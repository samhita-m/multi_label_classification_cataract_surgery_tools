# summer_internship_2017
Tool Annotation for Cataract Surgery
  
This project is one fof the grand challenges from https://grand-challenge.org/ and is implemented in the following steps.

There are 25 videos out of which 20 are assigned for training and the rest for testing. This is done randomly.
Frames are then extracted from these videos after which they are sectioned randomly to be given as input to the neural networks.
ResNet, AlexNet, DenseNet, GoogLeNet, VGG were attempted but only ResNet18 (20 epochs), ResNet34 (50 epochs), AlexNet (20 epochs) are implemented and analysis is done on accuracy. Other networks required much more compututaional space than available then. 
    
