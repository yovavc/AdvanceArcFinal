from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import wandb

from RNN2 import RNNAttention, GPU

# Set the backend to 'soundfile'
torchaudio.set_audio_backend("soundfile")

# Login to wandb with the API key
wandb.login(key="8c46208fe9e553bdaa921b70d41ec7601e302cce")

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'random', 'grid'
    'metric': {
        'name': 'Test Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01, 0.1]
        },
        'betas': {
            'values': [(0.9, 0.999), (0.95, 0.999), (0.9, 0.98)]
        },
        'eps': {
            'values': [1e-8, 1e-7, 1e-6]
        },
        'epochs': {
            'values': [10, 20, 30]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='kws-project-biu')


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

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


device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

batch_size = 256

train_set = SubsetSC("training")
valid_set = SubsetSC("validation")
test_set = SubsetSC("testing")

transform = MelSpectrogram(16000)
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
    with wandb.init(config=config):
        config = wandb.config

        transformer = RNNAttention(35)
        transformer.to(device)

        optimizer = optim.Adam(transformer.parameters(),
                               lr=config.learning_rate,
                               betas=config.betas,
                               eps=config.eps)
        loss_fn = torch.nn.CrossEntropyLoss()

        pbar_update = 1 / (len(train_loader) + len(test_loader))

        for epoch in range(1, config.epochs + 1):
            transformer.train()
            count = 0
            total_loss = 0

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}") as pbar:
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    target = target.to(device)
                    data = data.to(device).squeeze()
                    data = transform(data)
                    output = transformer(data)
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

            transformer.eval()
            correct = 0
            total_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    data = transform(data).squeeze()
                    output = transformer(data)
                    loss = loss_fn(output, target)
                    total_loss += loss.item()

                    predicted = torch.argmax(output, 1)
                    correct += number_of_correct(predicted, target)
                    pbar.update(pbar_update)

            avg_loss = total_loss / len(test_loader)
            accuracy = correct / len(test_loader.dataset)

            wandb.log({"Test Loss": avg_loss, "Test Accuracy": accuracy})

            print(f" - Test Epoch: {epoch}\tLoss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")


# Start the sweep
wandb.agent(sweep_id, train, count=10)
