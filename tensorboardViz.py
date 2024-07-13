import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from RNN1 import RNNAttention  # Adjust import based on your model file

# Define your model
model = RNNAttention(35)

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(256, 128, 81)  # Adjust the shape based on your model's expected input

# Initialize TensorBoard writer
writer = SummaryWriter('runs/model_visualization')

# Add model graph to TensorBoard
writer.add_graph(model, dummy_input)

print("Model graph has been added to TensorBoard.")
writer.close()
