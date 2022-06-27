from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import Sequential
from torch import flatten

class CNN(Module):
    def __init__(self, numberOfChannels, numberOfClasses, batch_size):
        # Get Module functionality
        super(CNN, self).__init__()

        # General parameters
        self.numberOfClasses = numberOfClasses
        self.batch_size = batch_size
        self.flatten_size = 14700

        # First convolutional layer
        self.convlayer1 = Sequential(
            Conv2d(in_channels=numberOfChannels, out_channels=6, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(5, 5), stride=(4, 4)) )

        # Second convolutional layer
        self.convlayer2 = Sequential(
            Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(5, 5), stride=(4, 4)) )

        # MLP layer
        self.MLPLayer = Sequential(
        Linear(in_features=self.flatten_size, out_features=1000),
        ReLU(),
        Linear(in_features=1000, out_features=250),
        ReLU(),
        Linear(in_features=250, out_features=self.numberOfClasses),
        LogSoftmax(dim=1) )

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)

        x = flatten(x, 1)

        x = self.MLPLayer(x)

        return x