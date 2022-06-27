from library.cnn_pytorch.cnn_handler import CNN_handler
from library.local_vit_pytorch.vit_handler import ViT_handler

import os

print("What algorithm would you like to use?")
print("(0) CNN")
print("(1) ViT")
x = input()

CNN_menu = ("\n(-1) Exit program\n"
+ "(0) Create new date CNN\n"
+ "(1) Create new format CNN\n"
+ "(2) Train date\n"
+ "(3) Train format\n"
+ "(4) Test date\n"
+ "(5) Test format\n"
+ "(6) Save model\n"
+ "(7) Load model\n")

ViT_menu = ("\n(-1) Exit program\n"
+ "(0) Create new date ViT\n"
+ "(1) Create new format ViT\n"
+ "(2) Train date\n"
+ "(3) Train format\n"
+ "(4) Test date\n"
+ "(5) Test format\n"
+ "(6) Save model\n"
+ "(7) Load model\n")

if (x == "0"):
    handler = CNN_handler(3, 21)
    classifier = 0

    # Do CNN stuff
    while (True):
        print(CNN_menu)
        x = input()

        if (x == "-1"):
            break
        elif (x == "0"):
            print("Creating new date CNN")
            handler = CNN_handler(3, 21)
        elif (x == "1"):
            print("Creating new format CNN")
        elif (x == "2"):
            print("Training CNN on date classification")
            handler.train_date()
        elif (x == "3"):
            print("Training CNN on format classification")
        elif (x == "4"):
            print("Testing CNN on date classification")
            handler.test_date()
        elif (x == "5"):
            print("Testing CNN on format classification")
        elif (x == "6"):
            print("Saving model. What should the filename be?")
            x = input()
            handler.save_model(x)
        elif (x == "7"):
            print("Loading model. What model should be loaded?")
            print("Currently saved models: ")
            print(os.listdir("saved_models/cnn_models/"))
            x = input()
            handler.load_model(x)
            print("Is this a (0) date classifier or a (1) format classifier?")
            x = input()
            classifier = int(x)
        else:
            print("Incorrect input")

elif (x == "1"):
    handler = ViT_handler(3, 21)
    classifier = 0
    # Do ViT stuff
    while (True):
        print(ViT_menu)
        x = input()

        if (x == "-1"):
            break
        elif (x == "0"):
            print("Creating new date ViT")
            handler = ViT_handler(3, 21)
        elif (x == "1"):
            print("Creating new format ViT")
            handler = ViT_handler(3, 21)
        elif (x == "2"):
            print("Training ViT on date classification")
            handler.training()
        elif (x == "3"):
            print("Training ViT on format classification")
            handler.train_date()
        elif (x == "4"):
            print("Testing ViT on date classification")
            handler.test_date()
        elif (x == "5"):
            print("Testing ViT on format classification")
            handler.test_date()
        elif (x == "6"):
            print("Saving model. What should the filename be?")
            x = input()
            handler.save_model(x)
        elif (x == "7"):
            print("Loading model. What model should be loaded?")
            print("Currently saved models: ")
            print(os.listdir("saved_models/vit_models/"))
            x = input()
            handler.load_model(x)
            print("Is this a (0) date classifier or a (1) format classifier?")
            x = input()
            classifier = int(x)
        else:
            print("Incorrect input")

else:
    print("Incorrect input")

print("Ending program...")
