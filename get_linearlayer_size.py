from library.cnn_pytorch.cnn import CNN
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchsize = 128
channels = 3

model = CNN(channels, 21, batchsize).to(device)
input = torch.FloatTensor(batchsize, channels, 600, 600)

model.eval()

with torch.no_grad():
    model(input)