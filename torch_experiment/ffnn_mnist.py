import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


# create a small fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size, **kwargs):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.ffnn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.ffnn(x)
        return logits


def check_accuracy(loader, model):
    correct_cnt, samples_cnt = 0, 0
    # need to tell the model that now we are evaluating
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)
            logits = model(x)
            predictions = logits.argmax(1)

            correct_cnt += (predictions == y).sum()
            samples_cnt += predictions.shape[0]
    model.train()
    return correct_cnt / samples_cnt

# %%
if __name__ == '__main__':
    # %%
    torch.manual_seed(0)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparams
    params = dict(
        input_size=28 * 28,
        num_classes=10,
        hidden_layer_size=256,
        batch_size=32,
        lr=0.0001,
        epochs=10,
        train_size=0.8,
        dev_size=0.2
    )

    # load data
    data = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
    test = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)

    # %%
    train, dev = torch.utils.data.random_split(data, [int(len(data) * params['train_size']), int(len(data) * params['dev_size'])])

    train_loader = DataLoader(dataset=train, batch_size=params['batch_size'], shuffle=True)
    dev_loader = DataLoader(dataset=dev, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=params['batch_size'], shuffle=True)

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train), size=(1,)).item()
    #     img, label = train[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # init network
    model = NN(**params).to(device)
    print(model)

    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    print(device)
    # train
    for epoch in range(params['epochs']):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # flatten the data, preserving the batch dimension
            data = data.reshape(data.shape[0], -1)

            # forward pass
            logits = model(data)
            loss = loss_fn(logits, targets)

            # backward
            # Call optimizer.zero_grad() to reset the gradients of model parameters.
            # Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
            optimizer.zero_grad()
            loss.backward()
            # gradient descent (adam) step
            optimizer.step()

        acc = check_accuracy(dev_loader, model)
        print(f'{epoch:3}) Accuracy val :: {acc:.3f}')

    # accuracy on test
    acc = check_accuracy(test_loader, model)
    print(f'Accuracy test :: {acc:.3f}')
