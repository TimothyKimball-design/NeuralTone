/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.computer.vision;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import javax.imageio.ImageIO;

public class NeuralNetworkForPictureEnchancement {

public static void main(String[] args) {
        try {
            String inputImagePath = "input.jpg";
            String targetImagePath = "target.jpg";
            String targetImage2Path = "target1.jpg";
            String targetImage3Path = "target2.jpg";
            String targetImage4Path = "target3.jpg";
            String targetImage5Path = "target4.jpg";
            String targetImage6Path = "target5.jpg";
            String targetImage7Path = "target6.jpg";
            String targetImage8Path = "target7.jpg";
            String targetImage9Path = "target8.jpg";
            String targetImage10Path = "target9.jpg";
            BufferedImage inputImage = ImageIO.read(new File(inputImagePath));
            BufferedImage targetImage = ImageIO.read(new File(targetImagePath));
            BufferedImage targetImage2 = ImageIO.read(new File (targetImage2Path));
            BufferedImage targetImage3 = ImageIO.read(new File (targetImage3Path));
            BufferedImage targetImage4 = ImageIO.read(new File (targetImage4Path));
            BufferedImage targetImage5 = ImageIO.read(new File (targetImage5Path));
            BufferedImage targetImage6 = ImageIO.read(new File(targetImage6Path));
            BufferedImage targetImage7 = ImageIO.read(new File (targetImage7Path));
            BufferedImage targetImage8 = ImageIO.read(new File (targetImage8Path));
            BufferedImage targetImage9 = ImageIO.read(new File (targetImage9Path));
            BufferedImage targetImage10 = ImageIO.read(new File (targetImage10Path));
            
            // Convert the input and target images to grayscale
            BufferedImage inputGrayscale = convertToGrayscale(inputImage);
            BufferedImage targetGrayscale = convertToGrayscale(targetImage);
            BufferedImage targetGrayscale2 = convertToGrayscale(targetImage2);
            BufferedImage targetGrayscale3 = convertToGrayscale(targetImage3);
            BufferedImage targetGrayscale4 = convertToGrayscale(targetImage4);
            BufferedImage targetGrayscale5 = convertToGrayscale(targetImage5);
            BufferedImage targetGrayscale6 = convertToGrayscale(targetImage6);
            BufferedImage targetGrayscale7 = convertToGrayscale(targetImage7);
            BufferedImage targetGrayscale8 = convertToGrayscale(targetImage8);
            BufferedImage targetGrayscale9 = convertToGrayscale(targetImage9);
            BufferedImage targetGrayscale10 = convertToGrayscale(targetImage10);
            
            // Prepare the training data
            double[][] trainingData = prepareTrainingData(inputGrayscale, targetGrayscale, targetGrayscale2, targetGrayscale3, targetGrayscale4, targetGrayscale5, targetGrayscale6, targetGrayscale7, targetGrayscale8, targetGrayscale9, targetGrayscale10);

            // Initialize and train the neural network
            int inputSize = trainingData[0].length;
            int hiddenSize = 64;
            int outputSize = 1;
            NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize);
            trainNeuralNetwork(neuralNetwork, trainingData);

            // Perform image enhancement using the trained neural network
            BufferedImage enhancedImage = enhanceImage(inputGrayscale, neuralNetwork);

            // Save the enhanced image
            String outputPath = "output.jpg";
            ImageIO.write(enhancedImage, "jpg", new File(outputPath));

            System.out.println("Image enhancement completed.");
        } catch (IOException e) {
        }
}

public static BufferedImage convertToGrayscale(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage grayscaleImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(image.getRGB(x, y));
                int luminance = (int) (color.getRed() * 0.299 + color.getGreen() * 0.587 + color.getBlue() * 0.114);
                Color grayscaleColor = new Color(luminance, luminance, luminance);
                grayscaleImage.setRGB(x, y, grayscaleColor.getRGB());
            }
        }

        return grayscaleImage;
}

public static double[][] prepareTrainingData(BufferedImage inputImage, BufferedImage targetImage, BufferedImage targetGrayscale2, BufferedImage targetGrayscale3, BufferedImage targetGrayscale4, BufferedImage targetGrayscale5, BufferedImage targetGrayscale6, BufferedImage targetGrayscale7, BufferedImage targetGrayscale8, BufferedImage targetGrayscale9, BufferedImage targetGrayscale10) {
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        double[][] trainingData = new double[width * height][2];

        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                trainingData[index][0] = (inputImage.getRGB(x, y) & 0xFF) / 255.0;
                trainingData[index][1] = (targetImage.getRGB(x, y) & 0xFF) / 255.0;

                index++;
            }
        }

        return trainingData;
}

public static void trainNeuralNetwork(NeuralNetwork neuralNetwork, double[][] trainingData) {
    int epochs = 10;  // Number of training epochs
    double learningRate = 10;  // Learning rate

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

        // Iterate over the training data
        for (double[] trainingData1 : trainingData) {
            double[] input = new double[]{trainingData1[0]};
            double[] target = new double[]{trainingData1[1]};
            // Perform a forward pass
            double[] output = neuralNetwork.predict(input);
            // Calculate the error
            double error = target[0] - output[0];
            totalError += Math.abs(error);
            // Perform the backpropagation algorithm
            double[] hiddenOutput = NeuralNetwork.predictHiddenLayer(input, neuralNetwork.weightsInputToHidden);
            neuralNetwork.updateWeights(hiddenOutput, output, target, learningRate);
        }

        // Print the average error for each epoch
        double averageError = totalError / trainingData.length;
        System.out.println("Epoch: " + epoch + ", Average Error: " + averageError);
    }
}


public static BufferedImage enhanceImage(BufferedImage image, NeuralNetwork neuralNetwork) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage enhancedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int input = image.getRGB(x, y) & 0xFF;
                double[] inputArray = {input};
                double[] outputArray = neuralNetwork.predict(inputArray);
                int output = (int) (outputArray[0] * 255);
                Color enhancedColor = new Color(output, output, output);
                enhancedImage.setRGB(x, y, enhancedColor.getRGB());
            }
        }

        return enhancedImage;
}

    
public static class NeuralNetwork {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double[][] weightsInputToHidden;
    private final double[][] weightsHiddenToOutput;

public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Initialize weights with random values
    weightsInputToHidden = initializeWeights(inputSize, hiddenSize);
    weightsHiddenToOutput = initializeWeights(hiddenSize, outputSize);
}

private static double[][] initializeWeights(int rows, int cols) {
    double[][] weights = new double[rows][cols];
    Random random = new Random();
    double weightScale = 1.0 / Math.sqrt(rows);  // Scale the weights based on the number of input connections

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i][j] = random.nextGaussian() * weightScale;
        }
    }

    return weights;
}

public void train(double[][] trainingData, int epochs, double learningRate) {
    // Neural network training code...
    // Implement your own neural network training algorithm here
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

        // Perform one epoch of training using the training data
        for (double[] data : trainingData) {
            double[] input = Arrays.copyOfRange(data, 0, inputSize);
            double[] target = Arrays.copyOfRange(data, inputSize, inputSize + outputSize);

            // Forward pass
            double[] hiddenOutput = predictHiddenLayer(input, weightsInputToHidden);
            double[] output = predictOutputLayer(hiddenOutput, weightsHiddenToOutput);

            // Calculate the error
            double[] error = new double[outputSize];
            for (int i = 0; i < outputSize; i++) {
                error[i] = target[i] - output[i];
                totalError += Math.pow(error[i], 2);
            }

            // Backpropagation
            // Update weights between hidden and output layers
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double delta = learningRate * error[j] * output[j] * (1.0 - output[j]) * hiddenOutput[i];
                    weightsHiddenToOutput[i][j] += delta;
                }
            }

            // Update weights between input and hidden layers
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    double delta = 0.0;
                    for (int k = 0; k < outputSize; k++) {
                        delta += error[k] * output[k] * (1.0 - output[k]) * weightsHiddenToOutput[j][k];
                    }
                    delta *= learningRate * hiddenOutput[j] * (1.0 - hiddenOutput[j]) * input[i];
                    weightsInputToHidden[i][j] += delta;
                }
            }
        }

        System.out.println("Epoch: " + epoch + ", Error: " + totalError);
    }
}

public double[] predict(double[] input) {
        // Neural network prediction code...
        // Implement your own neural network prediction logic here

        // Forward pass
        double[] hiddenOutput = predictHiddenLayer(input, weightsInputToHidden);
        double[] output = predictOutputLayer(hiddenOutput, weightsHiddenToOutput);

        return output;
}

 
private static double[] predictHiddenLayer(double[] input, double[][] weightsInputToHidden) {
        // Perform the forward pass for the hidden layer
        // Implement your own forward pass logic here
        double[] hiddenOutput = new double[weightsInputToHidden[0].length];
        for (int i = 0; i < hiddenOutput.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < input.length; j++) {
                sum += input[j] * weightsInputToHidden[j][i];
            }
            hiddenOutput[i] = activationFunction(sum);
        }
        return hiddenOutput;
    }

private static double[] predictOutputLayer(double[] hiddenOutput, double[][] weightsHiddenToOutput) {
        // Perform the forward pass for the output layer
        // Implement your own forward pass logic here
        double[] output = new double[weightsHiddenToOutput[0].length];
        for (int i = 0; i < output.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenOutput.length; j++) {
                sum += hiddenOutput[j] * weightsHiddenToOutput[j][i];
            }
            output[i] = activationFunction(sum);
        }
        return output;
}

    // Activation function (e.g., sigmoid, ReLU, etc.)
private static double activationFunction(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
}

private static void trainNeuralNetwork(NeuralNetwork neuralNetwork, double[][] trainingData, int epoch, double learningRate) {
    // Backpropagation training code...
    // Implement your own backpropagation algorithm here
    
    // Iterate over each training data sample
    for (double[] data : trainingData) {
        double[] input = Arrays.copyOfRange(data, 0, neuralNetwork.inputSize);
        double[] target = Arrays.copyOfRange(data, neuralNetwork.inputSize, neuralNetwork.inputSize + neuralNetwork.outputSize);

        // Forward pass
        double[] hiddenOutput = NeuralNetwork.predictHiddenLayer(input, neuralNetwork.weightsInputToHidden);
        double[] output = NeuralNetwork.predictOutputLayer(hiddenOutput, neuralNetwork.weightsHiddenToOutput);

        // Backward pass
        double[] outputError = calculateOutputError(output, target);
        double[] hiddenError = calculateHiddenError(hiddenOutput, outputError, neuralNetwork.weightsHiddenToOutput);

        // Update weights
        neuralNetwork.updateWeights(hiddenOutput, output, target, learningRate);
        neuralNetwork.updateHiddenWeights(input, hiddenError, learningRate);
    }
}

private static double[] calculateOutputError(double[] output, double[] target) {
    // Calculate the error between output and target
    // Implement your own error calculation method here
    double[] error = new double[output.length];
    for (int i = 0; i < output.length; i++) {
        error[i] = target[i] - output[i];
    }
    return error;
    }

private static double[] calculateHiddenError(double[] hiddenOutput, double[] outputError, double[][] weightsHiddenToOutput) {
    // Calculate the error in the hidden layer
    // Implement your own hidden error calculation method here
    double[] error = new double[hiddenOutput.length];
    for (int i = 0; i < hiddenOutput.length; i++) {
        double sum = 0.0;
        for (int j = 0; j < outputError.length; j++) {
            sum += outputError[j] * weightsHiddenToOutput[i][j];
        }
        error[i] = sum * hiddenOutput[i] * (1 - hiddenOutput[i]);
    }
    return error;
}

private void updateWeights(double[] hiddenOutput, double[] output, double[] target, double learningRate) {
    // Update the weights between the hidden and output layers
    // Implement your own weight update method here
    for (int i = 0; i < output.length; i++) {
        for (int j = 0; j < hiddenOutput.length; j++) {
            double weightUpdate = learningRate * output[i] * (1 - output[i]) * (target[i] - output[i]) * hiddenOutput[j];
            weightsHiddenToOutput[j][i] += weightUpdate;
        }
    }
}

private void updateHiddenWeights(double[] input, double[] hiddenError, double learningRate) {
    // Update the weights between the input and hidden layers
    // Implement your own weight update method here
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            double weightUpdate = learningRate * hiddenError[i] * input[j];
            weightsInputToHidden[j][i] += weightUpdate;
        }
    }
}

}

}

