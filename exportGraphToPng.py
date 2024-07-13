import torch
from torchviz import make_dot
from RNN1 import RNNAttention
# Define your model
model = RNNAttention(35)

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(256, 128, 81)  # Adjust the shape based on your model's expected input

# Forward pass to get the output
output = model(dummy_input)

# Visualize the model
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('model_visualization')

print("Model visualization created.")
