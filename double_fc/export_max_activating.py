import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------
# Model Definition (SmallNet)
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
# Function to Generate Max-Activation Image with Sparsity Encouragement
# ---------------------
def get_max_activation_image(model, unit, device='cpu', lr=0.1, steps=100, lambda_sparse=0.001):
    """
    Generates a 28x28 image that maximizes the activation for the specified unit (0-63)
    in the fc1 layer while encouraging sparsity (i.e. fewer pixels on).
    
    The loss is:
    
         Loss = -activation[0, unit] + lambda_sparse * sum(x)
    
    After optimization, the image is binarized (threshold=0.5).
    """
    # Start with a random 28x28 image with values in [0,1]
    img = np.random.uniform(0.3, 0.7, (28, 28)).astype(np.float32)
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    x.requires_grad = True
    optimizer = optim.Adam([x], lr=lr)

    # Set up a hook to capture fc1 activations.
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output  # shape: (1, 64)
    hook = model.fc1.register_forward_hook(hook_fn)

    for step in range(steps):
        optimizer.zero_grad()
        model(x)  # forward pass (hook updates "activation")
        # Loss: maximize activation and add sparsity penalty (sum(x))
        loss = -activation[0, unit] + lambda_sparse * torch.sum(x)
        loss.backward()
        optimizer.step()
        x.data.clamp_(0, 1)
    hook.remove()

    # Binarize the optimized image (threshold=0.5)
    x_bin = (x > 0.5).float()
    # Return a numpy array of shape (28,28) with values 0 or 1
    return x_bin.detach().cpu().squeeze().numpy()

# ---------------------
# Helper Function to Pack a 28x28 Binary Image into 98 Bytes
# ---------------------
def pack_image_to_bytes(image):
    """
    Packs a 28x28 numpy array of 0s and 1s (784 bits) into a list of 98 bytes.
    The packing is done in row-major order, 8 pixels per byte (MSB is the first pixel).
    """
    flat = image.flatten()  # shape (784,)
    bytes_list = []
    for i in range(0, len(flat), 8):
        byte = 0
        for j in range(8):
            bit = int(flat[i+j])
            byte |= (bit << (7 - j))
        bytes_list.append(byte)
    return bytes_list  # list of 98 integers (0-255)

# ---------------------
# Export Function: Write a 2D C Array of 64 Images (each 98 bytes) to a Header File
# ---------------------
def export_activations_to_c(packed_images, filename="activations.h"):
    """
    Exports a list of 64 packed images (each a list of 98 bytes) as a 2D C array.
    The array is defined as:
    
      const uint8_t max_activation_images[64][98] PROGMEM = { ... };
    """
    with open(filename, "w") as f:
        f.write("// Auto-generated max activation images for fc1 units\n")
        f.write("// Each image is a 28x28 binary image packed into 98 bytes (8 pixels per byte)\n\n")
        f.write("const uint8_t max_activation_images[64][98] PROGMEM = {\n")
        for idx, bytes_list in enumerate(packed_images):
            f.write("    { ")
            for j, b in enumerate(bytes_list):
                f.write(f"0x{b:02X}")
                if j != len(bytes_list)-1:
                    f.write(", ")
            f.write(" }")
            if idx != len(packed_images)-1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n")
    print(f"Export complete. {filename} now contains a 64x98 uint8_t array.")

# ---------------------
# Main: Generate, Export, and Stitch Max-Activation Images for All 64 fc1 Units
# ---------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SmallNet()
model.load_state_dict(torch.load('small_mnist_cnn_binarized.pt', map_location=device))
model.eval()

raw_images = []     # will store each 28x28 raw binary image (numpy array)
packed_images = []  # will store the corresponding 98-byte packed version

# You can adjust lambda_sparse to change how strongly sparsity is encouraged.
lambda_sparse = 0.01

for unit in range(64):
    print(f"Generating max activation image for fc1 unit {unit} with lambda_sparse={lambda_sparse} ...")
    img = get_max_activation_image(model, unit, device=device, lr=0.1, steps=100, lambda_sparse=lambda_sparse)
    raw_images.append(img)
    packed = pack_image_to_bytes(img)
    assert len(packed) == 98, "Packed image must have 98 bytes."
    packed_images.append(packed)

# Export the packed images as a C header file
export_activations_to_c(packed_images, filename="activations.h")

# ---------------------
# Stitch the 64 raw images together into an 8x8 grid and display
# ---------------------
grid_rows = 8
grid_cols = 8
grid_width = grid_cols * 28
grid_height = grid_rows * 28

grid_image = Image.new('L', (grid_width, grid_height), color=255)

for idx, img in enumerate(raw_images):
    # Convert each binary image (0 or 1) to displayable format (0 or 255)
    pil_im = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    row = idx // grid_cols
    col = idx % grid_cols
    grid_image.paste(pil_im, (col * 28, row * 28))

plt.figure(figsize=(8, 8))
plt.imshow(grid_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

grid_image.save("fc1_64_units_grid.jpg")
print("Stitched grid saved as 'fc1_64_units_grid.jpg'.")
