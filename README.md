# NeuralTone

This project shows how to use neural networks to enhance images. I accomplish this by leveraging a neural network implementation in Java that has been trained to improve grayscale images.

Overview

A set of target images is used to train the neural network, and the trained model is then applied to improve an input image. To increase overall performance, multiple neural networks are trained simultaneously during the training process.

Training Data: A set of target images (art2.jpg to art10.jpg) is used to prepare the training data.

Neural Network: Three neural networks are constructed and trained in parallel.

Enhancement: Using the trained neural networks, the input image is made better.

Output Image: enhanced.jpg is where the improved image is saved.

Configuration

You can tweak the neural network architecture, training parameters, and other settings in the NeuralNetworkForPictureEnchancement.java file.

Neural Network Architecture

The neural network architecture consists of an input layer, a hidden layer, and an output layer. The number of nodes in the hidden layer is configurable.

License

This project is licensed under the MIT License - see the LICENSE file for details.

