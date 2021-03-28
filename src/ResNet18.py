import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data
from torchvision.models import resnet18
from torchvision import datasets, transforms


#  ResNet Class Definition
class ResNet18(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchSize = 64
        self.loss = nn.CrossEntropyLoss()
        self.testDL = torch.utils.data.DataLoader(
            datasets.MNIST('./',
                           train=False,
                           download=False,
                           transform=transforms.Compose([transforms.ToTensor(), ])),
            self.batchSize,
            shuffle=False)

    @auto_move_data
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)

    def predict(self, model: pl.LightningModule, data):
        model.freeze()
        probs = torch.softmax(model(data))
        preds = torch.argmax(probs, dim=1)
        return preds, probs
