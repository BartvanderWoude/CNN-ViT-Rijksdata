from library.cnn_pytorch.cnn import CNN
from library.dataset.date_dataset import DateDataset

from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import numpy as np
import torch
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_handler:
    def __init__(self, channels, classes):
        # Basic parameters
        self.NUM_CHANNELS = channels
        self.NUM_CLASSES = classes
        self.BATCH_SIZE = 64
        self.INITIAL_LR = 1e-3
        self.EPOCHS = 1

        # Initialize an untrained model
        self.model = CNN(self.NUM_CHANNELS, self.NUM_CLASSES, self.BATCH_SIZE).to(device)

        # Initialize and split DataLoader
        # load the KMNIST dataset as placeholder
        data = DateDataset("data/date/")        
        
        (self.trainData, self.testData) = random_split(data,
	                                        [len(data) - 3000, 3000],
	                                        generator=torch.Generator().manual_seed(int(time.time())))
        (self.trainData, self.valData) = random_split(self.trainData,
                                            [len(self.trainData) - 1000, 1000],
                                            generator=torch.Generator().manual_seed(int(time.time()) + 1))
        
        self.trainDataLoader = DataLoader(self.trainData, shuffle=True, batch_size=self.BATCH_SIZE)
        self.valDataLoader = DataLoader(self.valData, batch_size=self.BATCH_SIZE)
        self.testDataLoader = DataLoader(self.testData, batch_size=self.BATCH_SIZE)

        self.trainSteps = len(self.trainDataLoader.dataset) // self.BATCH_SIZE
        self.valSteps = len(self.valDataLoader.dataset) // self.BATCH_SIZE

    
    def train_date(self):
        starttime = time.time()

        # initialize our optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=self.INITIAL_LR)
        lossFunction = nn.NLLLoss()

        # initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        ######################################################################
        # loop over our epochs
        for e in range(0, self.EPOCHS):
            # set the model in training mode
            self.model.train()

            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0

            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0

            counter = 0

            # loop over the training set
            for (x, y) in self.trainDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = lossFunction(pred, y)

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                counter = counter + 1
                print("Train batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.trainData)), end="\r")
                if counter % self.BATCH_SIZE*10 == 0:
                    self.save_model("temp")                
            
            print("Finished training loop")
            
            # switch off autograd for evaluation
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()

                counter = 0

                # loop over the validation set
                for (x, y) in self.valDataLoader:
                    # send the input to the device
                    (x, y) = (x.to(device), y.to(device))

                    # make the predictions and calculate the validation loss
                    pred = self.model(x)
                    totalValLoss += lossFunction(pred, y)

                    # calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                    counter = counter + 1
                    print("Validation batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.valData)), end="\r")
            
                # calculate the average training and validation loss
                avgTrainLoss = totalTrainLoss / self.trainSteps
                avgValLoss = totalValLoss / self.valSteps

                # calculate the training and validation accuracy
                trainCorrect = trainCorrect / len(self.trainDataLoader.dataset)
                valCorrect = valCorrect / len(self.valDataLoader.dataset)

                # update our training history
                H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
                H["train_acc"].append(trainCorrect)
                H["val_loss"].append(avgValLoss.cpu().detach().numpy())
                H["val_acc"].append(valCorrect)

                # print the model training and validation information
                print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
                print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
                print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))
            
        self.save_model(str(time.time()))

        endtime = time.time()

        print("Elapsed time: " + str(endtime - starttime))
        
    def test_date(self):
        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            self.model.eval()
            
            # initialize a list to store our predictions
            totalTestCorrect = 0

            counter = 0

            # loop over the test set
            for (x, y) in self.testDataLoader:
                # send the input to the device
                x = x.to(device)

                # make the predictions and add them to the list
                pred = self.model(x)
                totalTestCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                counter = counter + 1
                print("Test batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.testData)), end="\r")


            # generate a classification report
            # print(classification_report(self.testData.targets.cpu().numpy(),
            #     np.array(predictions), target_names=self.testData.classes))
            totalTestCorrect = totalTestCorrect / len(self.testDataLoader.dataset)
            print("Test accuracy: " + str(totalTestCorrect))
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), "saved_models/cnn_models/" + filename)
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load("saved_models/cnn_models/" + filename))
