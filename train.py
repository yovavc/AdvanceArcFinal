from random import random  # Import the random function from the random module

from torchaudio.datasets import SPEECHCOMMANDS  # Import the SPEECHCOMMANDS dataset from torchaudio
import torch  # Import the PyTorch library
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.optim as optim  # Import optimization algorithms from PyTorch
import torchaudio  # Import torchaudio for audio processing
import os  # Import the os module for operating system interactions
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking  # Import specific audio transforms
from tqdm import tqdm  # Import tqdm for progress bars
import wandb  # Import wandb for experiment tracking
import torchaudio.transforms as T  # Import torchaudio transforms with a shorthand alias
import random  # Import the random module for random operations
from EncoderCNN import UNetEncoder  # Import the UNetEncoder class from a local module

# Set the backend for torchaudio to 'soundfile' for audio processing
torchaudio.set_audio_backend("soundfile")

# Login to wandb with the API key for experiment tracking
wandb.login(key="8c46208fe9e553bdaa921b70d41ec7601e302cce")

# Define the configuration for the hyperparameter sweep
sweep_config = {
    'method': 'grid',  # Use grid search for hyperparameter optimization
    'metric': {
        'name': 'Max Test Accuracy',  # The metric to optimize is Max Test Accuracy
        'goal': 'maximize'  # The goal is to maximize this metric
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001]  # A fixed learning rate for all configurations
        },
        'betas': {
            'values': [(0.95, 0.999)]  # Beta values for the Adam optimizer
        },
        'eps': {
            'values': [1e-8]  # Epsilon value for the Adam optimizer
        },
        'epochs': {
            'values': [30]  # Train for 30 epochs
        },
        'time_mask_param': {
            'values': [60]  # Parameter for time masking augmentation
        },
        'start_filters': {
            'values': [1, 2, 4, 8]  # Different starting filter sizes for the model
        },
        'dropOutProbability': {
            'values': [0]  # Dropout probability for regularization
        },
        'freq_mask_param': {
            'values': [40]  # Parameter for frequency masking augmentation
        },
        'run_count': {
            'values': list(range(1, 100))  # Run the experiment for different counts
        }
    }
}

# Initialize the sweep with the configuration
sweep_id = wandb.sweep(sweep_config, project='kws-project-biu')


class Augmentations:
    def __init__(self):
        # Define a list of augmentations to apply to the audio data
        self.augmentations = [
            T.Vol(0.5),  # Volume adjustment
            T.PitchShift(sample_rate=16000, n_steps=4),  # Pitch shift
            T.FrequencyMasking(freq_mask_param=30),  # Frequency masking
            # T.TimeMasking(time_mask_param=50),  # Time masking (commented out)
        ]

    def __call__(self, waveform):
        # Apply augmentations with a certain probability
        for augment in self.augmentations:
            if random.random() < 0.5:  # 50% chance to apply each augmentation
                waveform = augment(waveform)
        return waveform  # Return the augmented waveform


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        # Initialize the SPEECHCOMMANDS dataset
        super().__init__("./", download=True)
        self.augment = Augmentations()  # Create an instance of the Augmentations class

        def load_list(filename):
            # Load a list of file paths from a text file
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        # Select the subset of data (training, validation, or testing)
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n):
        # Get an item from the dataset and apply augmentation
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        waveform = self.augment(waveform)  # Apply augmentations to the waveform
        return waveform, sample_rate, label  # Return the augmented data


# Define the list of labels for the dataset
labels = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
    'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree',
    'two', 'up', 'visual', 'wow', 'yes', 'zero'
]


def get_likely_index(tensor):
    # Get the index of the maximum value in a tensor
    return tensor.argmax(dim=1)


def idx2lbl(x):
    # Convert an index to a label
    return labels[x]


def lbl2idx(x):
    # Convert a label to its index
    return torch.tensor(labels.index(x))


def pad_sequence(batch):
    # Pad sequences in a batch to have the same length
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # Collate function to combine individual samples into a batch
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [lbl2idx(label)]
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


# Determine the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    num_workers = 2  # Use 2 workers for data loading on GPU
    pin_memory = True  # Pin memory to speed up data transfer to GPU
else:
    num_workers = 0  # No additional workers for CPU
    pin_memory = False  # No need to pin memory for CPU

batch_size = 256  # Define the batch size for training

# Initialize datasets for training, validation, and testing
train_set = SubsetSC("training")
valid_set = SubsetSC("validation")
test_set = SubsetSC("testing")

# Define a MelSpectrogram transform and move it to the device
transform = MelSpectrogram(sample_rate=16000)
transform.to(device)

# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


def number_of_correct(pred, target):
    # Calculate the number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def train(config=None):
    num_classes = 35  # Define the number of classes for classification
    max_test_accuracy = 0  # Track the maximum test accuracy
    with wandb.init(config=config, reinit=True):
        config = wandb.config
        tm = TimeMasking(time_mask_param=config.time_mask_param)  # Initialize time masking
        fm = FrequencyMasking(freq_mask_param=config.freq_mask_param)  # Initialize frequency masking
        model = UNetEncoder(num_classes=num_classes, start_filters=config.start_filters,
                            dropOutProbability=config.dropOutProbability)
        model.to(device)  # Move the model to the device

        # Initialize the Adam optimizer with specified parameters
        optimizer = optim.Adam(model.parameters(),
                               lr=config.learning_rate,
                               betas=config.betas,
                               eps=config.eps,
                               weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()  # Define the loss function

        pbar_update = 1 / (len(train_loader) + len(test_loader))  # Calculate progress bar update step

        for epoch in range(1, config.epochs + 1):
            model.train()  # Set the model to training mode
            count = 0  # Initialize correct prediction count

            total_loss = 0  # Initialize the total loss

            # Initialize the progress bar for the epoch
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}") as pbar:
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()  # Zero the gradients
                    target = target.to(device)  # Move the target labels to the device
                    data = data.to(device).squeeze()  # Move the input data to the device and remove singleton dimensions

                    # Apply time masking augmentation with a certain probability
                    if random.random() < 0.5:
                        data = tm(data)

                    # Apply the MelSpectrogram transform
                    data = transform(data)

                    # Apply frequency masking augmentation with a certain probability
                    if random.random() < 0.5:
                        data = fm(data)

                    # Add a singleton dimension for batch processing and adjust dimensions for the model
                    data = data.unsqueeze(0)
                    data = data.permute(1, 0, 2, 3)

                    # Forward pass through the model
                    output = model(data)
                    loss = loss_fn(output, target)  # Calculate the loss
                    loss.backward()  # Backward pass (compute gradients)
                    optimizer.step()  # Update model parameters

                    total_loss += loss.item()  # Accumulate the loss
                    predicted = torch.argmax(output, 1)  # Get the index of the max log-probability
                    count += number_of_correct(predicted, target)  # Count correct predictions

                    pbar.update(1)  # Update the progress bar
                    pbar.set_postfix(loss=loss.item())  # Display the current loss on the progress bar

            avg_loss = total_loss / len(train_loader)  # Calculate average loss for the epoch
            accuracy = count / len(train_loader.dataset)  # Calculate training accuracy

            # Log training loss and accuracy to wandb
            wandb.log({"Train Loss": avg_loss, "Train Accuracy": accuracy})

            print(f" - Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")

            model.eval()  # Set the model to evaluation mode
            correct = 0  # Initialize the correct prediction count for validation
            total_loss = 0  # Initialize the total loss for validation
            with torch.no_grad():  # Disable gradient calculation
                for data, target in test_loader:
                    data = data.to(device)  # Move the input data to the device
                    target = target.to(device)  # Move the target labels to the device
                    data = transform(data).squeeze()  # Apply the MelSpectrogram transform and remove singleton dimensions
                    data = data.unsqueeze(0)  # Add a singleton dimension for batch processing
                    data = data.permute(1, 0, 2, 3)  # Adjust dimensions for the model
                    output = model(data)  # Forward pass through the model
                    loss = loss_fn(output, target)  # Calculate the loss
                    total_loss += loss.item()  # Accumulate the loss

                    predicted = torch.argmax(output, 1)  # Get the index of the max log-probability
                    correct += number_of_correct(predicted, target)  # Count correct predictions

            avg_loss = total_loss / len(test_loader)  # Calculate average loss for the validation set
            accuracy = correct / len(test_loader.dataset)  # Calculate validation accuracy

            # Log validation loss and accuracy to wandb
            wandb.log({"Test Loss": avg_loss, "Test Accuracy": accuracy})

            print(f" - Test Epoch: {epoch}\tLoss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")

            # Check for early stopping
            if accuracy > max_test_accuracy:
                max_test_accuracy = accuracy  # Update the maximum test accuracy

                # Save the model locally
                torch.save(model.state_dict(), "model.pth")

                # Create an artifact to save the model to wandb
                artifact = wandb.Artifact("my-model_" + str(epoch), type="model")

                # Add the model file to the artifact
                artifact.add_file("model.pth")

                # Log the artifact to wandb
                wandb.log_artifact(artifact)
            elif accuracy < max_test_accuracy * 0.9:
                print(f"Early stopping at epoch {epoch} due to drop in test accuracy.")
                break  # Stop training early if the accuracy drops significantly
            wandb.log({"Max Test Accuracy": max_test_accuracy})  # Log the maximum test accuracy


# Start the sweep with the specified configuration and run the train function for each configuration
wandb.agent(sweep_id, train, count=6000)
