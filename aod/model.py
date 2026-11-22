import torch
import torch.nn as nn
import torch.nn.functional as F


class AODnet(nn.Module):

    def __init__(self):
        super(AODnet, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 3, 1, stride=1, padding=0
        )

        self.conv2 = nn.Conv2d(
            3, 3, 3, stride=1, padding=1
        )

        self.conv3 = nn.Conv2d(
            6, 3, 5, stride=1, padding=2
        )

        self.conv4 = nn.Conv2d(
            6, 3, 7, stride=1, padding=3
        )

        self.conv5 = nn.Conv2d(
            12, 3, 3, stride=1, padding=1
        )

        self.b = 1

    def forward(self, x):

        x1 = F.relu(self.conv1(x))

        x2 = F.relu(self.conv2(x1))

        cat1 = torch.cat(
            (x1, x2), 1
        )

        x3 = F.relu(
            self.conv3(cat1)
        )

        cat2 = torch.cat(
            (x2, x3), 1
        )

        x4 = F.relu(
            self.conv4(cat2)
        )

        cat3 = torch.cat(
            (x1, x2, x3, x4), 1
        )

        k = F.relu(
            self.conv5(cat3)
        )

        if k.size() != x.size():
            raise Exception(
                "k and input image have different sizes"
            )

        output = k * x - k + self.b

        return F.relu(output)