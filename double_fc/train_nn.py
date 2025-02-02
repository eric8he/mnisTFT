import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

# ---------------------
# Example Binarize Transform
# ---------------------
class BinarizeTransform(object):
    """
    A custom transform that thresholds the input (already converted to tensor)
    at a specified value, converting it to 0 or 1.
    """
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def __call__(self, img_tensor):
        # img_tensor is assumed to be in [0,1], shape [1, 28, 28]
        return (img_tensor > self.threshold).float()

# ---------------------
# Random Binarize Transform
# ---------------------
class RandomBinarizeTransform(object):
    """
    A custom transform that thresholds the input at a random value within
    a specified range, converting it to 0 or 1.
    """
    def __init__(self, min_threshold=0.1, max_threshold=0.3):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def __call__(self, img_tensor):
        # Randomly choose threshold for this call
        threshold = self.min_threshold + torch.rand(1).item() * (self.max_threshold - self.min_threshold)
        return (img_tensor > threshold).float()

# ---------------------
# Model (Same as Before)
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
# Create Datasets
# ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    RandomBinarizeTransform(min_threshold=0.1, max_threshold=0.3),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(28, scale=(0.95, 1.05))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ---------------------
# Train & Evaluate
# ---------------------
model = SmallNet()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train
model.train()
num_epochs = 20
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

# Evaluate
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100.0 * correct / len(test_loader.dataset)
print(f"\nTestset: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# (Optional) Save the trained model
torch.save(model.state_dict(), 'small_mnist_cnn_binarized.pt')


# ---------------------------------------------------
# Helper: Export an image example to weights.h
# ---------------------------------------------------
# --- ADDED OR MODIFIED CODE HERE ---
def export_example_to_c(image_tensor, label, pred_label, filename="weights.h", array_name="example_correct"):
    """
    Exports a single 28x28 binarized image + label info to the same C header file.
    image_tensor: 1 x 28 x 28 (FloatTensor) with values 0 or 1
    label: ground truth label (int)
    pred_label: predicted label (int)
    array_name: base name for the arrays
    """
    # image_tensor is shape [1, 28, 28]
    # Convert to numpy
    image_np = image_tensor.squeeze().cpu().numpy().astype(np.uint8)  # shape [28, 28]

    with open(filename, "a") as f:
        f.write(f"// Example: {array_name}\n")
        f.write(f"// Ground truth label = {label}, predicted = {pred_label}\n")
        f.write(f"const uint8_t {array_name}[] PROGMEM = {{\n")

        # Write out 28*28 = 784 values in row-major order
        idx = 0
        for row in range(28):
            f.write("    ")
            for col in range(28):
                val = image_np[row, col]
                f.write(f"{val}, ")
                idx += 1
            f.write("\n")
        f.write("};\n\n")


# ---------------------------------------------------
# Quantize + Export to 16-bit uint16_t arrays
# ---------------------------------------------------
def export_to_c_uint16(model, filename="weights.h", scale=256.0):
    """
    Exports model weights to a C file as uint16_t arrays in PROGMEM.
    The approach:
    1) Multiply each float by `scale`.
    2) Cast to int16, clamp to [-32768,32767].
    3) Store as uint16_t by offsetting with 65536 if negative.
    4) In inference, read as uint16_t, convert back to int16, then to float by dividing by `scale`.
    """
    state_dict = model.state_dict()

    with open(filename, "w") as f:
        f.write('// Auto-generated - DO NOT EDIT\n\n')

        # We'll store the scale as a constant if you want to reference it in Arduino code
        f.write(f'// Quantization scale used: {scale}\n')
        f.write(f'const float WEIGHT_SCALE = {scale}f;\n\n')

        for param_name, param_tensor in state_dict.items():
            param_data = param_tensor.cpu().numpy().flatten()
            shape_str = ' x '.join(map(str, param_tensor.shape))
            f.write(f'// {param_name}, shape: {shape_str}\n')
            c_name = param_name.replace('.', '_')

            f.write(f'const uint16_t {c_name}[] PROGMEM = {{\n')

            # For each value, quantize and clamp
            for idx, val in enumerate(param_data):
                # 1) scale
                scaled = val * scale
                # 2) round to nearest int
                i16_val = int(np.round(scaled))
                # 3) clamp
                i16_val = max(min(i16_val, 32767), -32768)
                # 4) convert int16 -> uint16_t
                u16_val = i16_val & 0xFFFF

                # Format multiple values per line
                if idx % 8 == 0:
                    f.write("    ")
                f.write(f"{u16_val}, ")
                if (idx + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n};\n\n")


# --- ADDED OR MODIFIED CODE HERE ---
# After we've exported weights, let's also export a correctly classified example.
# If you also want a misclassified example, you can do the same approach
# but checking for pred != target.
export_to_c_uint16(model, "weights.h", scale=256.0)

# Now find a correct example from the test set and export it.
model.eval()
found_correct = False
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True).squeeze(1)  # shape [batch_size]
        # For each item in the batch
        for i in range(len(target)):
            if pred[i].item() == target[i].item():
                # Found a correct example
                correct_data = data[i]           # shape [1, 28, 28]
                correct_label = target[i].item() # ground truth
                pred_label = pred[i].item()
                export_example_to_c(
                    image_tensor=correct_data,
                    label=correct_label,
                    pred_label=pred_label,
                    filename="weights.h",
                    array_name="example_correct"
                )
                found_correct = True
                break
        if found_correct:
            break

print("weights.h generated with model weights and one correctly classified example.")
