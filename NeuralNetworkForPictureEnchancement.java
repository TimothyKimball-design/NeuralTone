//Made by Tim



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
            
           
            double[][] trainingData = prepareTrainingData(inputGrayscale, targetGrayscale, targetGrayscale2, targetGrayscale3, targetGrayscale4, targetGrayscale5, targetGrayscale6, targetGrayscale7, targetGrayscale8, targetGrayscale9, targetGrayscale10);

          
            int inputSize = trainingData[0].length;
            int hiddenSize = 64;
            int outputSize = 1;
            NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize);
            trainNeuralNetwork(neuralNetwork, trainingData);

            
            BufferedImage enhancedImage = enhanceImage(inputGrayscale, neuralNetwork);

           
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
    int epochs = 10;  
    double learningRate = 10;  

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

      
        for (double[] trainingData1 : trainingData) {
            double[] input = new double[]{trainingData1[0]};
            double[] target = new double[]{trainingData1[1]};
            
            double[] output = neuralNetwork.predict(input);
            
            double error = target[0] - output[0];
            totalError += Math.abs(error);
           
            double[] hiddenOutput = NeuralNetwork.predictHiddenLayer(input, neuralNetwork.weightsInputToHidden);
            neuralNetwork.updateWeights(hiddenOutput, output, target, learningRate);
        }

       
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
    double weightScale = 1.0 / Math.sqrt(rows);  

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i][j] = random.nextGaussian() * weightScale;
        }
    }

    return weights;
}

public void train(double[][] trainingData, int epochs, double learningRate) {
   
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

        
        for (double[] data : trainingData) {
            double[] input = Arrays.copyOfRange(data, 0, inputSize);
            double[] target = Arrays.copyOfRange(data, inputSize, inputSize + outputSize);

            
            double[] hiddenOutput = predictHiddenLayer(input, weightsInputToHidden);
            double[] output = predictOutputLayer(hiddenOutput, weightsHiddenToOutput);

            double[] error = new double[outputSize];
            for (int i = 0; i < outputSize; i++) {
                error[i] = target[i] - output[i];
                totalError += Math.pow(error[i], 2);
            }

            
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double delta = learningRate * error[j] * output[j] * (1.0 - output[j]) * hiddenOutput[i];
                    weightsHiddenToOutput[i][j] += delta;
                }
            }

            
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
       

        
        double[] hiddenOutput = predictHiddenLayer(input, weightsInputToHidden);
        double[] output = predictOutputLayer(hiddenOutput, weightsHiddenToOutput);

        return output;
}

 
private static double[] predictHiddenLayer(double[] input, double[][] weightsInputToHidden) {
        
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

   
private static double activationFunction(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
}

private static void trainNeuralNetwork(NeuralNetwork neuralNetwork, double[][] trainingData, int epoch, double learningRate) {
   
    
   
    for (double[] data : trainingData) {
        double[] input = Arrays.copyOfRange(data, 0, neuralNetwork.inputSize);
        double[] target = Arrays.copyOfRange(data, neuralNetwork.inputSize, neuralNetwork.inputSize + neuralNetwork.outputSize);

       
        double[] hiddenOutput = NeuralNetwork.predictHiddenLayer(input, neuralNetwork.weightsInputToHidden);
        double[] output = NeuralNetwork.predictOutputLayer(hiddenOutput, neuralNetwork.weightsHiddenToOutput);

       
        double[] outputError = calculateOutputError(output, target);
        double[] hiddenError = calculateHiddenError(hiddenOutput, outputError, neuralNetwork.weightsHiddenToOutput);

       
        neuralNetwork.updateWeights(hiddenOutput, output, target, learningRate);
        neuralNetwork.updateHiddenWeights(input, hiddenError, learningRate);
    }
}

private static double[] calculateOutputError(double[] output, double[] target) {
   
    double[] error = new double[output.length];
    for (int i = 0; i < output.length; i++) {
        error[i] = target[i] - output[i];
    }
    return error;
    }

private static double[] calculateHiddenError(double[] hiddenOutput, double[] outputError, double[][] weightsHiddenToOutput) {
   
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
    
    for (int i = 0; i < output.length; i++) {
        for (int j = 0; j < hiddenOutput.length; j++) {
            double weightUpdate = learningRate * output[i] * (1 - output[i]) * (target[i] - output[i]) * hiddenOutput[j];
            weightsHiddenToOutput[j][i] += weightUpdate;
        }
    }
}

private void updateHiddenWeights(double[] input, double[] hiddenError, double learningRate) {
    
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            double weightUpdate = learningRate * hiddenError[i] * input[j];
            weightsInputToHidden[j][i] += weightUpdate;
        }
    }
}

}

}

