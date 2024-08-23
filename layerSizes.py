import torch
import torch.nn as nn
import torch.nn.functional as F
from EncoderCNN import UNetEncoder

num_classes= 35

# Instantiate the model
model = UNetEncoder(num_classes=num_classes, start_filters=32)

# Dictionary to store layer sizes
layer_sizes = {}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Hook function to capture the output size
def hook_fn(module, input, output):
    layer_sizes[module] = output.size()

# Register hooks for each layer
hooks = []
for layer in model.children():
    hooks.append(layer.register_forward_hook(hook_fn))

# Create a dummy input tensor
dummy_input = torch.randn(1, 1, 128, 81)  # Batch size of 1, 1 channel, 28x28 image

print(f"Number of parameters: {count_parameters(model)}")
# Run a forward pass to trigger the hooks
output = model(dummy_input)

# Print the captured layer sizes
for layer, size in layer_sizes.items():
    print(f"Layer: {layer}")
    print(f"Output size: {size}")
    print("-" * 20)

# Remove hooks
for hook in hooks:
    hook.remove()
