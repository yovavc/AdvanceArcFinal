from random import random

from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
from tqdm import tqdm
import wandb
import torchaudio.transforms as T
import random
from EncoderCNN import UNetEncoder

# Set the backend to 'soundfile'
torchaudio.set_audio_backend("soundfile")

# Login to wandb with the API key
wandb.login(key="8c46208fe9e553bdaa921b70d41ec7601e302cce")

# Define the sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'Max Test Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001]
        },
        'betas': {
            'values': [(0.95, 0.999)]
        },
        'eps': {
            'values': [1e-8]
        },
        'epochs': {
            'values': [30]
        },
        'time_mask_param': {
            'values': [10, 20]
        },
        'start_filters': {
            'values': [1, 2]

        },
        'dropOutProbability': {
            'values': [0]
        },
        'freq_mask_param': {
            'values': [40]
        },
        'run_count':{
            'values':  list(range(1, 301))
        }

    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='kws-project-biu')


class Augmentations:
    def __init__(self):
        self.augmentations = [
            T.Vol(0.5),
            T.PitchShift(sample_rate=16000, n_steps=4),
            # Remove TimeStretch as it requires a fixed rate
            T.FrequencyMasking(freq_mask_param=30),
            # T.TimeMasking(time_mask_param=50),
        ]


    def __call__(self, waveform):
        for augment in self.augmentations:
            if random.random() < 0:  # 50% chance to apply each augmentation
                waveform = augment(waveform)
        return waveform


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)
        self.augment = Augmentations()

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        waveform = self.augment(waveform)
        return waveform, sample_rate, label


labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy',
          'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
          'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


def get_likely_index(tensor):
    return tensor.argmax(dim=1)


def idx2lbl(x):
    return labels[x]


def lbl2idx(x):
    return torch.tensor(labels.index(x))


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [lbl2idx(label)]
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    num_workers = 2
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

batch_size = 256

train_set = SubsetSC("training")
valid_set = SubsetSC("validation")
test_set = SubsetSC("testing")

transform = MelSpectrogram(sample_rate=16000)
transform.to(device)

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
    return pred.squeeze().eq(target).sum().item()


def train(config=None):
    num_classes = 35
    max_test_accuracy = 0  # Track the maximum test accuracy
    with wandb.init(config=config, reinit=True):
        config = wandb.config
        tm = TimeMasking(time_mask_param=config.time_mask_param)
        fm = FrequencyMasking(freq_mask_param=config.freq_mask_param)
        model = UNetEncoder(num_classes=num_classes, start_filters=16, dropOutProbability=config.dropOutProbability)
        model.to(device)

        optimizer = optim.Adam(model.parameters(),
                               lr=config.learning_rate,
                               betas=config.betas,
                               eps=config.eps,
                               weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        pbar_update = 1 / (len(train_loader) + len(test_loader))

        for epoch in range(1, config.epochs + 1):
            model.train()
            count = 0
            total_loss = 0

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}") as pbar:
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    target = target.to(device)
                    data = data.to(device).squeeze()
                    if random.random() < 0:  # 50% chance to apply each augmentation
                        data = tm(data)

                    data = transform(data)

                    if random.random() < 0:  # 50% chance to apply each augmentation
                        data = fm(data)
                    data = data.unsqueeze(0)
                    data = data.permute(1, 0, 2, 3)

                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    predicted = torch.argmax(output, 1)
                    count += number_of_correct(predicted, target)

                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            accuracy = count / len(train_loader.dataset)

            wandb.log({"Train Loss": avg_loss, "Train Accuracy": accuracy})

            print(f" - Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")

            model.eval()
            correct = 0
            total_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    data = transform(data).squeeze()
                    data = data.unsqueeze(0)
                    data = data.permute(1, 0, 2, 3)
                    output = model(data)
                    loss = loss_fn(output, target)
                    total_loss += loss.item()

                    predicted = torch.argmax(output, 1)
                    correct += number_of_correct(predicted, target)
                    pbar.update(pbar_update)

            avg_loss = total_loss / len(test_loader)
            accuracy = correct / len(test_loader.dataset)

            wandb.log({"Test Loss": avg_loss, "Test Accuracy": accuracy})

            print(f" - Test Epoch: {epoch}\tLoss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")

            # Check for early stopping
            if accuracy > max_test_accuracy:
                max_test_accuracy = accuracy
                # Save the model locally
                torch.save(model.state_dict(), "model.pth")

                # Create an artifact
                artifact = wandb.Artifact("my-model_" + str(epoch), type="model")

                # Add the model file to the artifact
                artifact.add_file("model.pth")
                # Log the artifact
                wandb.log_artifact(artifact)
            elif accuracy < max_test_accuracy * 0.9:
                print(f"Early stopping at epoch {epoch} due to drop in test accuracy.")
                break
            wandb.log({"Max Test Accuracy": max_test_accuracy})


# Start the sweep
wandb.agent(sweep_id, train, count=6000)
