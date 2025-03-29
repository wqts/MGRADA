import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def get_batch_indices(dset, batch_size):
    batch_indices = torch.randint(0, len(dset['label']), (batch_size,), device="cuda")
    return batch_indices

def sharpen(probabilities, T):

    if probabilities.ndim == 1:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
            tempered
            / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )

    else:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered

def test(config, dset, model, feature):
    dataloader = DataLoader(TensorDataset(dset[feature], dset['label']), batch_size=1024)
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            logits = pred["logits"]
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_accuracy = correct / size
    return test_loss, test_accuracy