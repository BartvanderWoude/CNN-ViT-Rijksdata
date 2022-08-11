from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import Sequential
from torch import flatten

class CNN(Module):
    def __init__(self):
        # Current classification task (initially date classification (0))
        self.classifier = 0

        # Get Module functionality
        super(CNN, self).__init__()

        # Layer size feeding into MLP heads (calculated using get_linearlayer_size.py)
        self.flatten_size = 14700

        # First convolutional layer
        self.convlayer1 = Sequential(
            Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(5, 5), stride=(4, 4)) )

        # Second convolutional layer
        self.convlayer2 = Sequential(
            Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(5, 5), stride=(4, 4)) )

        # MLP layer
        self.MLP_date = Sequential(
        Linear(in_features=self.flatten_size, out_features=1000),
        ReLU(),
        Linear(in_features=1000, out_features=250),
        ReLU(),
        Linear(in_features=250, out_features=21),
        LogSoftmax(dim=1) )

        self.MLP_format = Sequential(
        Linear(in_features=self.flatten_size, out_features=1000),
        ReLU(),
        Linear(in_features=1000, out_features=250),
        ReLU(),
        Linear(in_features=250, out_features=55) )

    def setToDateClassification(self):
        self.classifier = 0
    
    def setToFormatClassification(self):
        self.classifier = 1

    def forward(self, x):
        # Perform concolutional layers
        x = self.convlayer1(x)
        x = self.convlayer2(x)

        # Flatten output matrix
        x = flatten(x, 1)

        # Based on which classification task perform the MLP head
        if (self.classifier == 0):
            x = self.MLP_date(x)
        else:
            x = self.MLP_format(x)

        return x