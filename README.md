# Digit Recognition System

## Description

This simple digit recognition system is an APP for detecting the hardwritten digits in the screen. It uses a convolution neural network (CNN). It allows users to write in various colors besides black. Besides, It is a convenient and easily accessible app, with a good-looking GUI built using tkinter with thorough tools, by which we can save the valuable result for subsequent analysis of differnet AI model. However, the RNN model currently used does not have very high accuracy for hardwritten digits recognition (esp, the number '1').

Note: The topics of this project are object-oriented programming, AI/ML and GUI.

## Features









## Requirement

* PC monitor with scale of 200% and display resolution of 2880x1920
* Python version: 3.9+
* Python libraries: `tkinter`, `cv2`, `numpy`, `PIL`, `tensorflow`, `os`, `screeninfo`, `matplotlib` and `keras`

## Usage

#### Training Model for Digit Recognition System

First, you need to train the CNN for the app. For this, please run `CNN_dian.py` directly. However, you could also skip this step and using your own model for recognition, but the model should be able to be used by `load_model` method in the `tensorflow.keras.models` module and used for the grayscale figure of type of `numpy.ndarray` with format of 1x28x28x1 (= \[# of samples\]x\[width\]x\[height\]x\[channels\]).

If you run `CNN_dian.py`, a png file named `myCNNPerformance.png` will be generated to show the model accuracy and loss for both train and test set. An example `myCNNPerformance.png` is shown below. From this figrue, the accuracy of our CNN model for test set is higher than 98%.

* CNN training takes around 10 min, please be patient.

![]()

#### Digit Recognition System











## Caveats

* Please make sure to set your PC monitor with scale of 200% and display resolution of 2880x1920. Other params of monitor may generate malfunction.
* Please make sure when hitting 'Recognize Digit(s)' button, the canvas wedgit in the root Tk window is not hidden by any object/pattern in your PC monitor window. Maximizing the Tk root window is recommanded!
* Please save the result figrue and table (with reconization data) mannually if you want them for later analysis. App would not save automatically for you.

## Future work
* Enable the result handwritten digits retrain the loaded model to get a more rubust and accurate model.
* Build a model with recognition of any ASCII character.
* Improve the accuracy of recognition of digit '1'.

## Reference

#### Articles and Guides for this Digit Recognition System
* [Handwritten Digit Recognition GUI App](https://medium.com/analytics-vidhya/handwritten-digit-recognition-gui-app-46e3d7b37287)

#### Convolution Neural Network
* [Convolutional Neural Networks, Explained](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)
* [How do convolutional neural networks work?](https://www.ibm.com/topics/convolutional-neural-networks)
* [Tensorflow - for beginners](https://www.tensorflow.org/tutorials)

#### Tkinter
* [Python Tkinter Tutorial](https://www.geeksforgeeks.org/python-tkinter-tutorial/?ref=lbp)
* [Tkinter - the Python interface for Tk](https://python-course.eu/tkinter/)
* [Build your own desktop apps with Python & Tkinter](https://www.pythonguis.com/tkinter-tutorial/#tkinter-getting-started)

#### OpenCV
* [OpenCV Python Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/?ref=lbp)

## Acknowledgements

* Many, many thanks to Anastasia Georgious from Johns Hopkins University for guidance of this project.
* Thanks to everyone who works on all the awesome Python machine learning and GUI libraries like tensorflow, cv2, tkinter, keras, etc, etc that makes this kind of stuff so easy and fun in Python.
