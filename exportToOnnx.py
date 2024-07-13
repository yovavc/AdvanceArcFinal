import torch
import torch.onnx
from EncoderCNN import UNetEncoder

# Define and load your model
model = UNetEncoder(num_classes= 35)
# model.load_state_dict(torch.load("file"))
model.eval()

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1,1, 40, 81)  # Adjust the shape based on your model's expected input

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'])

print("Model has been converted to ONNX format.")
