import onnx
import onnxruntime as ort
import numpy as np


def load_onnx_model(model_path):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)

    # Check that the IR is well-formed
    onnx.checker.check_model(onnx_model)

    return onnx_model


def print_layer_info(onnx_model):
    # Print layer details
    print("Layer Information:")
    for node in onnx_model.graph.node:
        print(f"Layer name: {node.name}")
        print(f"Op type: {node.op_type}")
        print("Inputs:")
        for inp in node.input:
            print(f"  {inp}")
        print("Outputs:")
        for out in node.output:
            print(f"  {out}")
        print("-" * 20)


def get_tensor_shapes(session, dummy_input):
    # Initialize a dictionary to store tensor shapes
    tensor_shapes = {}

    # Run inference to get output tensor shapes
    intermediate_layer_outputs = [output.name for output in session.get_outputs()]
    output_tensors = session.run(intermediate_layer_outputs, {session.get_inputs()[0].name: dummy_input})

    # Collect tensor shapes
    for name, tensor in zip(intermediate_layer_outputs, output_tensors):
        tensor_shapes[name] = tensor.shape

    return tensor_shapes


def main(model_path):
    # Load the ONNX model
    onnx_model = load_onnx_model(model_path)

    # Print layer information
    print_layer_info(onnx_model)

    # Create an inference session
    session = ort.InferenceSession(onnx_model.SerializeToString())

    # Get model input shape
    input_shape = session.get_inputs()[0].shape
    print(f"Expected input shape: {input_shape}")

    # Create a dummy input for shape inference
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Get tensor shapes
    tensor_shapes = get_tensor_shapes(session, dummy_input)

    # Print tensor shapes
    print("Tensor Shapes:")
    for tensor_name, shape in tensor_shapes.items():
        print(f"Tensor name: {tensor_name}, shape: {shape}")


if __name__ == "__main__":
    # Path to the ONNX model file
    model_path = "model.onnx"

    main(model_path)
