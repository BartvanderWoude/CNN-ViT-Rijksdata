from library.cnn_pytorch.cnn_handler import CNN_handler
from library.local_vit_pytorch.vit_handler import ViT_handler

import os

print("What algorithm would you like to use?")
print("(0) CNN")
print("(1) ViT")
x = input()

CNN_menu = ("\n(-1) Exit program\n"
+ "(0) Create new CNN\n"
+ "(1) Train date\n"
+ "(2) Train format\n"
+ "(3) Test date\n"
+ "(4) Test format\n"
+ "(5) Save model\n"
+ "(6) Load model\n")

ViT_menu = ("\n(-1) Exit program\n"
+ "(0) Create new ViT\n"
+ "(1) Train date\n"
+ "(2) Train format\n"
+ "(3) Test date\n"
+ "(4) Test format\n"
+ "(5) Save model\n"
+ "(6) Load model\n")

if (x == "0"):
    handler = CNN_handler()

    # Do CNN stuff
    while (True):
        print(CNN_menu)
        x = input()

        if (x == "-1"):
            break
        elif (x == "0"):
            print("Creating new CNN")
            handler = CNN_handler()
        elif (x == "1"):
            print("Training CNN on date classification")
            handler.train_date()
        elif (x == "2"):
            print("Training CNN on format classification")
            handler.train_format()
        elif (x == "3"):
            print("Testing CNN on date classification")
            handler.test_date()
        elif (x == "4"):
            print("Testing CNN on format classification")
            handler.test_format()
        elif (x == "5"):
            print("Saving model. What should the filename be?")
            x = input()
            handler.save_model(x)
        elif (x == "6"):
            print("Loading model. What model should be loaded?")
            print("Currently saved models: ")
            print(os.listdir("saved_models/cnn_models/"))
            x = input()
            handler.load_model(x)
        else:
            print("Incorrect input")

elif (x == "1"):
    handler = ViT_handler()

    # Do ViT stuff
    while (True):
        print(ViT_menu)
        x = input()

        if (x == "-1"):
            break
        elif (x == "0"):
            print("Creating new ViT")
            handler = ViT_handler()
        elif (x == "1"):
            print("Training ViT on date classification")
            handler.train_date()
        elif (x == "2"):
            print("Training ViT on format classification")
            handler.train_format()
        elif (x == "3"):
            print("Testing ViT on date classification")
            handler.test_date()
        elif (x == "4"):
            print("Testing ViT on format classification")
            handler.test_format()
        elif (x == "5"):
            print("Saving model. What should the filename be?")
            x = input()
            handler.save_model(x)
        elif (x == "6"):
            print("Loading model. What model should be loaded?")
            print("Currently saved models: ")
            print(os.listdir("saved_models/vit_models/"))
            x = input()
            handler.load_model(x)
        else:
            print("Incorrect input")

else:
    print("Incorrect input")

print("Ending program...")
