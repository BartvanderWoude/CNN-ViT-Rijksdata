from library.local_vit_pytorch.vit import ViT
from library.dataset.date_dataset import DateDataset
from library.dataset.format_dataset import FormatDataset

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch import nn
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViT_handler:
    def __init__(self):
        # Basic parameters
        self.BATCH_SIZE = 20
        self.EPOCHS = 430
        self.INITIAL_LR = 1e-3
        self.GAMMA = 0.7

        self.channels = 3
        self.image_size = 600
        self.patch_size = 50
        self.dim = 64
        self.depth = 5
        self.heads = 5
        self.mlp_dim = 128
        self.dropout = 0
        self.emb_dropout = 0

        # Initialize untrained model
        self.model = ViT(channels = self.channels,
                            image_size = self.image_size,
                            patch_size = self.patch_size,
                            dim = self.dim,
                            depth = self.depth,
                            heads = self.heads,
                            mlp_dim = self.mlp_dim,
                            dropout = self.dropout,
                            emb_dropout = self.emb_dropout).to(device)

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

#################### Date classification ####################
    def train_date(self):
        # Set classification task to date classification
        self.model.setToDateClassification()

        starttime = time.time()

        # Initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        # Loop over epochs
        for e in range(0, self.EPOCHS):
            # Perform epoch training
            (t_loss, t_acc, v_loss, v_acc) = self.train_date_epoch()

            # Update the training history
            H["train_loss"].append(t_loss.cpu().detach().numpy())
            H["train_acc"].append(t_acc)
            H["val_loss"].append(v_loss.cpu().detach().numpy())
            H["val_acc"].append(v_acc)

            # Give user feedback on training and validation
            print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(t_loss, t_acc))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(v_loss, v_acc))
        
        # Save the trained model
        self.save_model(str(time.time()))
        df = pd.DataFrame.from_dict(H)
        df.to_csv (r'data/training/vit_date_train.csv', index = False, header=True)

        endtime = time.time()
        print("Elapsed time: " + str(endtime - starttime))
    
    def train_date_epoch(self):
        # Reshuffle the data (due to memory issues it is loaded here again)
        self.trainDateDataLoader = DataLoader(self.trainDateData, shuffle=True, batch_size=self.BATCH_SIZE)

        # Initialize loss function and optimizer
        lossFunction = nn.NLLLoss()
        optimizer = Adam(self.model.parameters(), lr=self.INITIAL_LR)

        ######################################################################
        # Set the model in training mode
        self.model.train()

        # Initialize variables to keep track of total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # Initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        # Counter keeps track of processed batches
        trainCounter = 0

        # Loop over the training set
        for (x, y) in self.trainDateDataLoader:
            # Send the input to the device
            (x, y) = (x.to(device), y.to(device))

            # Perform a forward pass and calculate the training loss
            pred = self.model(x)
            loss = lossFunction(pred, y)

            # Zero out the gradients, perform the backpropagation step,
            # and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Update the counter
            trainCounter = trainCounter + 1

            # Give user feedback and save model every 10 batches
            print("Train batch: " + str(trainCounter * self.BATCH_SIZE) + "/" + str(len(self.trainDateData)), end="\r")
            if trainCounter % 10 == 0:
                self.save_model("temp")
            
            # Break it off after 20 batches due to memory issues
            if trainCounter >= 20:
                break
            
        # Switch off autograd for evaluation
        with torch.no_grad():
            # Set the model in evaluation mode
            self.model.eval()

            # Counter keeps track of number of processed batches
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
            avgTrainLoss = totalTrainLoss / (trainCounter * self.BATCH_SIZE)
            avgValLoss = totalValLoss / self.valDateSteps

            # Calculate the training and validation accuracy
            trainCorrect = trainCorrect / (trainCounter * self.BATCH_SIZE)
            valCorrect = valCorrect / len(self.valDateDataLoader.dataset)

        print("")

        return (avgTrainLoss, trainCorrect, avgValLoss, valCorrect)

    def test_date(self):
        # Set to date classification
        self.model.setToDateClassification()

        # Turn off autograd for testing evaluation
        with torch.no_grad():
            # Set the model in evaluation mode
            self.model.eval()
            
            # totalTestCorrect keeps track of correct predictions
            totalTestCorrect = 0

            # Counter keeps track of the number of processed batches
            counter = 0

            # Loop over the test set
            for (x, y) in self.testDateDataLoader:
                # Send the input to the device
                x = x.to(device)

                # Make the prediction and determine its correctness
                pred = self.model(x)
                totalTestCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                #Update the counter
                counter = counter + 1

                # Give user feedback
                print("Test batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.testDateData)), end="\r")

            # Give user feedback on test accuracy
            totalTestCorrect = totalTestCorrect / len(self.testDateDataLoader.dataset)
            print("Test accuracy: " + str(totalTestCorrect))

#################### Format classification ####################
    def train_format(self):
        # Set classification task to format classification
        self.model.setToFormatClassification()

        starttime = time.time()

        # Initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_positive_acc": [],
            "train_negative_acc": [],
            "val_loss": [],
            "val_positive_acc": [],
            "val_negative_acc": []
        }
        
        # Loop over epochs
        for e in range(0, self.EPOCHS):
            # Perform epoch training
            (t_loss, t_pos_acc, t_neg_acc, v_loss, v_pos_acc, v_neg_acc) = self.train_format_epoch()

            # Update the training history
            H["train_loss"].append(t_loss.cpu().detach().numpy())
            H["train_positive_acc"].append(t_pos_acc)
            H["train_negative_acc"].append(t_neg_acc)
            H["val_loss"].append(v_loss.cpu().detach().numpy())
            H["val_positive_acc"].append(v_pos_acc)
            H["val_negative_acc"].append(v_neg_acc)

            # Give user feedback on training and validation
            print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
            print("Train loss: {:.6f}, Train positive accuracy: {:.4f}, Train negative accuracy: {:.4f}".format(t_loss, t_pos_acc, t_neg_acc))
            print("Val loss: {:.6f}, Val positive accuracy: {:.4f}, Val negative accuracy: {:.4f}\n".format(v_loss, v_pos_acc, v_neg_acc))
        
        # Save trained model and training data
        self.save_model(str(time.time()))
        df = pd.DataFrame.from_dict(H)
        df.to_csv (r'data/training/vit_format_train.csv', index = False, header=True)

        endtime = time.time()
        print("Elapsed time: " + str(endtime - starttime))
    
    def train_format_epoch(self):
        # Reshuffle the data (due to memory issues it is loaded here again)
        self.trainFormatDataLoader = DataLoader(self.trainFormatData, shuffle=True, batch_size=self.BATCH_SIZE)

        # Initialize loss function and optimizer
        lossFunction = nn.BCEWithLogitsLoss()
        optimizer = Adam(self.model.parameters(), lr=self.INITIAL_LR)

        ######################################################################
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

        # Counter keeps track of number of processed batches
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

            # Update the counter
            counter = counter + 1

            # Give user feedback and save model every 10 batches
            print("Train batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.trainFormatData)), end="\r")
            if counter % 10 == 0:
                self.save_model("temp")
            
            # Break it off after 20 batches due to memory issues
            if counter >= 20:
                break
        
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

            # Counter keeps track of the number of processed batches
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

                # Update the counter
                counter = counter + 1

                # Give user feedback
                print("Validation batch: " + str(counter * self.BATCH_SIZE) + "/" + str(len(self.valFormatData)), end="\r")
        
            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / self.trainFormatSteps
            avgValLoss = totalValLoss / self.valFormatSteps

            # Calculate the validation accuracy
            valPositiveAccuracy = correctPositive / allPositive
            valNegativeAccuracy = correctNegative / allNegative

        print("")

        return (avgTrainLoss, trainPositiveAccuracy, trainNegativeAccuracy, avgValLoss, valPositiveAccuracy, valNegativeAccuracy)

    def test_format(self):
        # Set classification to format classification
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

                # Analyze which labels the ViT classifies the input
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

#################### Saving/loading models ####################
    def save_model(self, filename):
        torch.save(self.model.state_dict(), "saved_models/vit_models/" + filename)
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load("saved_models/vit_models/" + filename))