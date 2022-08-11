from library.cnn_pytorch.cnn import CNN
from library.dataset.date_dataset import DateDataset
from library.dataset.format_dataset import FormatDataset

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_handler:
    def __init__(self):
        # Basic parameters
        self.BATCH_SIZE = 64
        self.INITIAL_LR = 1e-3
        self.EPOCHS = 1

        # Initialize an untrained model
        self.model = CNN().to(device)

        # Initialize and split DataLoader
        # Load the date dataset
        dateData = DateDataset("data/rijksdata/")        
        
        (self.trainDateData, self.testDateData) = random_split(dateData,
	                                        [len(dateData) - 3000, 3000],
	                                        generator=torch.Generator().manual_seed(int(time.time())))
        (self.trainDateData, self.valDateData) = random_split(self.trainDateData,
                                            [len(self.trainDateData) - 1000, 1000],
                                            generator=torch.Generator().manual_seed(int(time.time()) + 1))
        
        self.trainDateDataLoader = DataLoader(self.trainDateData, shuffle=True, batch_size=self.BATCH_SIZE)
        self.valDateDataLoader = DataLoader(self.valDateData, batch_size=self.BATCH_SIZE)
        self.testDateDataLoader = DataLoader(self.testDateData, batch_size=self.BATCH_SIZE)

        self.trainDateSteps = len(self.trainDateDataLoader.dataset) // self.BATCH_SIZE
        self.valDateSteps = len(self.valDateDataLoader.dataset) // self.BATCH_SIZE

        # Load the format dataset
        formatData = FormatDataset("data/rijksdata/")        
        
        (self.trainFormatData, self.testFormatData) = random_split(formatData,
	                                        [len(formatData) - 3000, 3000],
	                                        generator=torch.Generator().manual_seed(int(time.time())))
        (self.trainFormatData, self.valFormatData) = random_split(self.trainFormatData,
                                            [len(self.trainFormatData) - 1000, 1000],
                                            generator=torch.Generator().manual_seed(int(time.time()) + 1))
        
        self.trainFormatDataLoader = DataLoader(self.trainFormatData, shuffle=True, batch_size=self.BATCH_SIZE)
        self.valFormatDataLoader = DataLoader(self.valFormatData, batch_size=self.BATCH_SIZE)
        self.testFormatDataLoader = DataLoader(self.testFormatData, batch_size=self.BATCH_SIZE)

        self.trainFormatSteps = len(self.trainFormatDataLoader.dataset) // self.BATCH_SIZE
        self.valFormatSteps = len(self.valFormatDataLoader.dataset) // self.BATCH_SIZE


########################### DATE ###########################
    def train_date(self):
        # Specify to model we're doing date classification
        self.model.setToDateClassification()

        starttime = time.time()

        # Initialize our optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=self.INITIAL_LR)
        lossFunction = nn.NLLLoss()

        # Initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        ######################################################################
        # Loop over our epochs
        for e in range(0, self.EPOCHS):
            # Set the model in training mode
            self.model.train()

            # Initialize variables for the sum of training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0

            # Initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0

            # Counter is used to keep track of number of processed batches
            counter = 0

            # Loop over the training set
            for (x, y) in self.trainDateDataLoader:
                # Send input to the device
                (x, y) = (x.to(device), y.to(device))

                # Perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = lossFunction(pred, y)

                # Zero out the gradients, perform the backpropagation step,
                # and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Ddd the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Update the counter
                counter = counter + 1

                print("Loss: " + str(totalTrainLoss / (counter*self.BATCH_SIZE)))
                # Give user feedback and save model every 10 batches
                print("Train batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.trainDateData)), end="\r")
                
                if counter % 10 == 0:
                    self.save_model("temp")                
            
            print("Finished training loop")
            
            # Switch off autograd for evaluation
            with torch.no_grad():
                # Set the model in evaluation mode
                self.model.eval()

                # Counter is used to keep track of number of processed batches
                counter = 0

                # Loop over the validation set
                for (x, y) in self.valDateDataLoader:
                    # Send the input to the device
                    (x, y) = (x.to(device), y.to(device))

                    # Make the predictions and calculate the validation loss
                    pred = self.model(x)
                    totalValLoss += lossFunction(pred, y)

                    # Calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                    # Update the counter
                    counter = counter + 1

                    # Give user feedback
                    print("Validation batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.valDateData)), end="\r")
            
                # Calculate the average training and validation loss
                avgTrainLoss = totalTrainLoss / self.trainDateSteps
                avgValLoss = totalValLoss / self.valDateSteps

                # Calculate the training and validation accuracy
                trainCorrect = trainCorrect / len(self.trainDateDataLoader.dataset)
                valCorrect = valCorrect / len(self.valDateDataLoader.dataset)

                # Add this epoch to training history
                H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
                H["train_acc"].append(trainCorrect)
                H["val_loss"].append(avgValLoss.cpu().detach().numpy())
                H["val_acc"].append(valCorrect)

                # Give user feedback on model training and validation information
                print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
                print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
                print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))
        
        # Saved the trained model and save the training history
        self.save_model(str(time.time()))
        df = pd.DataFrame.from_dict(H)
        df.to_csv (r'data/training/cnn_date_train.csv', index = False, header=True)

        endtime = time.time()
        print("Elapsed time: " + str(endtime - starttime))
        
    def test_date(self):
        # Specify to model we're doing date classification
        self.model.setToDateClassification()

        # Turn off autograd for testing evaluation
        with torch.no_grad():
            # Set the model in evaluation mode
            self.model.eval()
            
            # Initialize a variable to keep track of number of correct predictions
            totalTestCorrect = 0

            # Counter is used to keep track of number of processed batches
            counter = 0

            # Loop over the test set
            for (x, y) in self.testDateDataLoader:
                # Send the input to the device
                x = x.to(device)

                # Make the predictions and evaluate how many were correct
                pred = self.model(x)
                totalTestCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Update counter
                counter = counter + 1

                # Give user feedback
                print("Test batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.testDateData)), end="\r")


            # Give accuracy report
            totalTestCorrect = totalTestCorrect / len(self.testDateDataLoader.dataset)
            print("Test accuracy: " + str(totalTestCorrect))
    
########################### FORMAT ###########################
    def train_format(self):
        # Specify to model we're doing format classification
        self.model.setToFormatClassification()

        starttime = time.time()

        # Initialize our optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=self.INITIAL_LR)
        lossFunction = nn.BCEWithLogitsLoss()

        # Initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_positive_acc": [],
            "train_negative_acc": [],
            "val_loss": [],
            "val_positive_acc": [],
            "val_negative_acc": []
        }

        ######################################################################
        # Loop over our epochs
        for e in range(0, self.EPOCHS):
            # Set the model in training mode
            self.model.train()

            # Initialize variables for the sum of training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0

            # Initialize the number of correct predictions in the training
            # and validation step
            allPositive = 0
            allNegative = 0
            correctPositive = 0
            correctNegative = 0

            # Counter is used to keep track of number of processed batches
            counter = 0

            # Loop over the training set
            for (x, y) in self.trainFormatDataLoader:
                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Perform a forward pass and calculate the training loss
                pred = self.model(x)
                pred, y = pred.double(), y.double()
                loss = lossFunction(pred, y)

                # Zero out the gradients, perform the backpropagation step,
                # and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Add the loss to the total training loss so far
                totalTrainLoss += loss

                # Analyze which labels the CNN classifies the input
                pred = torch.where(pred > 0.0, 1., 0.)

                # Out of target positives, calculate how many it got right
                allPositive += y.type(torch.float).sum().item()
                y_positive = torch.where(y > 0.5, 1., 3.)
                correctPositive += torch.eq(pred, y_positive).type(torch.float).sum().item()

                # Out of target negatives, calculate how many it got right
                allNegative += (torch.where(y < 0.5, 1., 0.)).type(torch.float).sum().item()
                y_negative = torch.where(y < 0.5, 0., 3.)
                correctNegative += torch.eq(pred, y_negative).type(torch.float).sum().item()

                # Update counter
                counter = counter + 1

                # Give user feedback and save trained model every 10 batches
                print("Train batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.trainFormatData)), end="\r")
                if counter % 10 == 0:
                    break
                    self.save_model("temp")            
            
            # Calculate true positive and true negative accuracy
            trainPositiveAccuracy = correctPositive / allPositive
            trainNegativeAccuracy = correctNegative / allNegative

            print("Finished training loop")
            
            # Reset for validation step
            allPositive = 0
            allNegative = 0
            correctPositive = 0
            correctNegative = 0
            
            # Switch off autograd for evaluation
            with torch.no_grad():
                # Set the model in evaluation mode
                self.model.eval()

                # Counter is used to keep track of number of processed batches
                counter = 0

                # Loop over the validation set
                for (x, y) in self.valFormatDataLoader:
                    # Send the input to the device
                    (x, y) = (x.to(device), y.to(device))

                    # Make the predictions and calculate the validation loss
                    pred = self.model(x)
                    pred, y = pred.double(), y.double()

                    # Add the loss to the total validation loss
                    totalValLoss += lossFunction(pred, y)

                    # Analyze which labels the CNN classifies the input
                    pred = torch.where(pred > 0.0, 1., 0.)

                    # Out of target positives, calculate how many it got right
                    allPositive += y.type(torch.float).sum().item()
                    y_positive = torch.where(y > 0.5, 1., 3.)
                    correctPositive += torch.eq(pred, y_positive).type(torch.float).sum().item()

                    # Out of target negatives, calculate how many it got right
                    allNegative += (torch.where(y < 0.5, 1., 0.)).type(torch.float).sum().item()
                    y_negative = torch.where(y < 0.5, 0., 3.)
                    correctNegative += torch.eq(pred, y_negative).type(torch.float).sum().item()

                    # Update counter
                    counter = counter + 1

                    # Give user feedback
                    print("Validation batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.valFormatData)), end="\r")
            
                # Calculate the average training and validation loss
                avgTrainLoss = totalTrainLoss / self.trainFormatSteps
                avgValLoss = totalValLoss / self.valFormatSteps

                # Calculate the validation accuracy
                valPositiveAccuracy = correctPositive / allPositive
                valNegativeAccuracy = correctNegative / allNegative

                # Update the training history
                H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
                H["train_positive_acc"].append(trainPositiveAccuracy)
                H["train_negative_acc"].append(trainNegativeAccuracy)
                H["val_loss"].append(avgValLoss.cpu().detach().numpy())
                H["val_positive_acc"].append(valPositiveAccuracy)
                H["val_negative_acc"].append(valNegativeAccuracy)

                # Give user feedback on training and validation
                print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
                print("Train loss: {:.6f}, Train positive accuracy: {:.4f}, Train negative accuracy: {:.4f}".format(avgTrainLoss, trainPositiveAccuracy, trainNegativeAccuracy))
                print("Val loss: {:.6f}, Val positive accuracy: {:.4f}, Val negative accuracy: {:.4f}\n".format(avgValLoss, valPositiveAccuracy, valNegativeAccuracy))
        
        # Save model after training and save training history
        self.save_model(str(time.time()))
        df = pd.DataFrame.from_dict(H)
        df.to_csv (r'data/training/cnn_format_train.csv', index = False, header=True)

        endtime = time.time()
        print("Elapsed time: " + str(endtime - starttime))
        
    def test_format(self):
        # Specify to model we're doing format classification
        self.model.setToFormatClassification()

        # Turn off autograd for testing evaluation
        with torch.no_grad():
            # Set the model in evaluation mode
            self.model.eval()
            
            # Initialize the number of correct predictions in the training
            # and validation step
            allPositive = 0
            allNegative = 0
            correctPositive = 0
            correctNegative = 0

            # Counter keeps track of the number of processed batches
            counter = 0

            # Loop over the test set
            for (x, y) in self.testFormatDataLoader:
                # Send the input to the device
                x = x.to(device)

                # Make the predictions
                pred = self.model(x)

                # Analyze which labels the CNN classifies the input
                pred = torch.where(pred > 0.0, 1., 0.)
                
                # Out of target positives, calculate how many it got right
                allPositive += y.type(torch.float).sum().item()
                y_positive = torch.where(y > 0.5, 1., 3.)
                correctPositive += torch.eq(pred, y_positive).type(torch.float).sum().item()

                # Out of target negatives, calculate how many it got right
                allNegative += (torch.where(y < 0.5, 1., 0.)).type(torch.float).sum().item()
                y_negative = torch.where(y < 0.5, 0., 3.)
                correctNegative += torch.eq(pred, y_negative).type(torch.float).sum().item()

                # Update the counter
                counter = counter + 1

                # Give user feedback
                print("Test batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.testFormatData)), end="\r")


            # Give accuracy report
            positiveAccuracy = correctPositive / allPositive
            print("Positive accuracy: " + str(positiveAccuracy))
            negativeAccuracy = correctNegative / allNegative
            print("Negative accuracy: " + str(negativeAccuracy))

    def save_model(self, filename):
        # Save model in folder 'saved_models/cnn_models/'
        torch.save(self.model.state_dict(), "saved_models/cnn_models/" + filename)
    
    def load_model(self, filename):
        # Load model with specified name from 'saved_models/cnn_models/'
        self.model.load_state_dict(torch.load("saved_models/cnn_models/" + filename))
