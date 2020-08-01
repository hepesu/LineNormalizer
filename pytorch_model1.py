import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from datagen import gen_data

SIZE = 128
ITERATIONS = 5100
BATCH_SIZE = 8
SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


def data_generator():
    rnd = np.random.RandomState(SEED)
    while True:
        raw, norm = gen_data(rnd, BATCH_SIZE)
        yield torch.from_numpy(raw).permute(0, 3, 1, 2), \
              torch.from_numpy(norm).permute(0, 3, 1, 2)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------
    #  Model and Optimizer
    # ----------
    model = Net().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.5, last_epoch=-1)

    # ----------
    #  Data
    # ----------
    dataIter = data_generator()

    # ----------
    #  Training
    # ----------
    for iteration in range(1, ITERATIONS + 1):
        rawData, normData = next(dataIter)

        rawData = rawData.to(DEVICE)
        normData = normData.to(DEVICE)

        optimizer.zero_grad()

        loss = F.mse_loss(model(rawData), normData)

        loss.backward()
        optimizer.step()

        print("[Iteration %d] [loss: %f]" % (iteration, loss.item()))

        if iteration % 100 == 0:
            torch.save(model.state_dict(), os.path.join('weight', '{}.pth'.format(iteration)))

        if iteration < 4000:
            scheduler.step()


if __name__ == "__main__":
    main()
