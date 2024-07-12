from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchaudio, torchvision
import torch.utils.data as utils_data
import sys
import os
from torchaudio.transforms import AddNoise, Vol, PitchShift, Speed
from torchvision.transforms import Compose

from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
# from transformer import Transformer
# from download import Trans
from RNN2 import RNNAttention, GPU
# Set the backend to 'soundfile'
torchaudio.set_audio_backend("soundfile")


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


labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
def idx2lbl(x):
    return labels[x]
def lbl2idx(x):
    return torch.tensor(labels.index(x))
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [lbl2idx(label)]

    # Group the list of tensors into a batched tensor
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
    pin_memory=pin_memory#,
    # transform=transform_rand
)
# print(train_loader[0])
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


transformer = RNNAttention(35)
transformer.load_state_dict(torch.load("file"))
transformer.eval()


transformer.cuda()
transformer.to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-7)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()


def train(model, epoch, log_interval):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        target = target.to(device)
        # apply transform and model on whole batch directly on device
        data = data.to(device).squeeze()
        data = transform(data)
        output = model(data)
        # print(output)
        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = loss_fn(output, target) # l
        loss.backward()
        optimizer.step()

        # print training stats

            # print(torch.argmax(output, 1))
            # print(target)
        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

        predicted = torch.argmax(output, 1)
        count += number_of_correct(predicted, target)
    # NOTICE: the loss printed here is not correct, do not use it, the correct value is though
    print(f" - Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}, correct: {count/ (len(train_loader)*batch_size)}")

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data).squeeze()

        output = model(data)

        predicted = torch.argmax(output, 1)
        correct += number_of_correct(predicted, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f" - Test Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


log_interval = 100
n_epoch = 10

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(transformer, epoch, log_interval)
        # scheduler.step()
    test(transformer, epoch)

def predict(tensor):
    # Use the model to predict the label of the waveform

    tensor = tensor.to(device)

    tensor = transform(tensor)
    tensor = transformer(tensor)
    tensor = get_likely_index(tensor)
    tensor = idx2lbl(tensor.squeeze())
    return tensor


waveform, sample_rate, utterance, *_ = train_set[-1]
# ipd.Audio(waveform.numpy(), rate=sample_rate)

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")


for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
    output = predict(waveform)
    if output != utterance:
        # ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
        break
else:
    print("All examples in this dataset were correctly classified!")
    print("In this case, let's just look at the last data point")
    # ipd.Audio(waveform.numpy(), rate=sample_rate)
    print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")

torch.save(transformer.state_dict(), "file")