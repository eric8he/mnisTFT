import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random

# ---------------------
# Define a custom binarization function with STE
# ---------------------
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Binarize with threshold 0.5.
        return (input > 0.5).float()
    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient through unchanged.
        return grad_output

# Convenience wrapper.
binarize_ste = BinarizeSTE.apply

# ---------------------
# Define the Custom MNIST Model
# ---------------------
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ---------------------
# Hook to capture activations from a layer
# ---------------------
class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output  # Capture the output from the layer.
    def close(self):
        self.hook.remove()

# ---------------------
# Helper function for Total Variation regularization
# ---------------------
def total_variation(x):
    """
    Compute total variation loss for a batch of images.
    x: tensor of shape (1, 1, H, W)
    """
    tv_h = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ---------------------
# Regularized & Binarized Visualizer for fc1 (layer_index=4) using STE
# ---------------------
class FilterVisualizerMNISTRegBinarizedSTE:
    def __init__(self, model, size=28, device='cpu'):
        """
        model: pretrained SmallNet model.
        size: input image size expected by the network (28 for MNIST).
        device: 'cpu' or 'cuda'.
        """
        self.size = size
        self.device = device
        self.model = model.to(device).eval()
        # Freeze model parameters.
        for param in self.model.parameters():
            param.requires_grad = False

    def visualize(self, layer_idx, unit_idx, lr=0.1, opt_steps=100, 
                  lambda_tv=0.001, lambda_l2=0.001, jitter=2):
        """
        Optimizes a random 28x28 image so that the activation for the chosen unit in fc1 is maximized,
        with added TV and L2 regularization, optional jitter, and with binarization via STE.
        The underlying variable remains continuous (so gradients flow) but is binarized in the forward pass.
        """
        sz = self.size
        # Start with a random image with values in [0,1].
        img = np.random.uniform(0.3, 0.7, (sz, sz)).astype(np.float32)

        # Attach hook to the target module (fc1).
        target_module = list(self.model.children())[layer_idx]
        activations = SaveFeatures(target_module)

        # Prepare the image tensor: shape (1, 1, sz, sz)
        x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
        x.requires_grad = True

        optimizer = torch.optim.Adam([x], lr=lr, weight_decay=1e-6)
        for step in range(opt_steps):
            optimizer.zero_grad()
            # Apply jitter: randomly shift the image a few pixels if jitter > 0.
            if jitter > 0:
                shift_x = random.randint(-jitter, jitter)
                shift_y = random.randint(-jitter, jitter)
                x_jit = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))
            else:
                x_jit = x

            # Instead of feeding x directly, feed its binarized version via our STE function.
            x_bin = binarize_ste(x_jit)
            self.model(x_bin)  # Forward pass; hook captures fc1 output.
            act = activations.features  # shape: [1, 64]
            # Primary loss: negative activation (to maximize activation).
            loss = -act[0, unit_idx]
            # L2 regularization on the continuous variable.
            loss = loss + lambda_l2 * torch.norm(x)
            # Total variation regularization.
            loss = loss + lambda_tv * total_variation(x)
            
            loss.backward()
            optimizer.step()
            # Clamp x to valid range [0,1] (continuous variable).
            x.data.clamp_(0, 1)
            if step % 20 == 0:
                print(f"Unit {unit_idx}, step {step}/{opt_steps}, Loss: {loss.item():.4f}")
        
        # At the end, produce the final output by binarizing x.
        self.output = binarize_ste(x).detach().cpu().squeeze().numpy()
        activations.close()

# ---------------------
# Main Code: Optimize binarized images for all 64 fc1 units with regularization using STE
# ---------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SmallNet()
model.load_state_dict(torch.load('small_mnist_cnn_binarized.pt', map_location=device))
model.eval()

layer_index = 4  # fc1 is at index 4
all_images = []
FV = FilterVisualizerMNISTRegBinarizedSTE(model, size=28, device=device)

for unit in range(64):
    print(f"\nOptimizing image for fc1 unit {unit}...")
    FV.visualize(layer_index, unit, lr=0.1, opt_steps=100, 
                 lambda_tv=0.001, lambda_l2=0.001, jitter=2)
    all_images.append(FV.output)

# ---------------------
# Overlay border and label directly on the 28x28 binarized images using PIL.
# ---------------------
font = ImageFont.load_default()

labeled_images = []
for i, img_arr in enumerate(all_images):
    # Convert the binarized image (values 0 or 1) to an 8-bit grayscale image.
    im = Image.fromarray(np.uint8(np.clip(img_arr, 0, 1) * 255), mode='L')
    draw = ImageDraw.Draw(im)
    # Draw a border (rectangle) on the image.
    draw.rectangle([(0, 0), (27, 27)], outline=0, width=1)
    # Prepare the label and draw it in the top-left corner.
    label = str(i)
    bbox = font.getbbox(label)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # Draw a small white rectangle as background for the label.
    draw.rectangle([(1, 1), (1 + text_w, 1 + text_h)], fill=255)
    draw.text((1, 1), label, fill=0, font=font)
    labeled_images.append(im)

# ---------------------
# Stitch the 64 labeled 28x28 images into an 8x8 grid.
# ---------------------
grid_cols = 8
grid_rows = 8
grid_width = grid_cols * 28
grid_height = grid_rows * 28

grid_image = Image.new('L', (grid_width, grid_height), color=255)
for idx, cell in enumerate(labeled_images):
    row = idx // grid_cols
    col = idx % grid_cols
    x = col * 28
    y = row * 28
    grid_image.paste(cell, (x, y))

# ---------------------
# Display and save the final grid.
# ---------------------
plt.figure(figsize=(8, 8))
plt.imshow(grid_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

grid_image.save("fc1_64_units_grid_28x28_binarized_STE.jpg")
print("Stitched grid saved as 'fc1_64_units_grid_28x28_binarized_STE.jpg'.")
