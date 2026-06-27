from torch import nn

from src.config import settings

class XRAYNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )

        self.block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=128*28*28,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.block5 = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.block6 = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=64,
                out_features=settings.NO_OF_CLASSES
            )
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = XRAYNet()