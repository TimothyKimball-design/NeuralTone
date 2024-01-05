//Made by Tim 

package com.mycompany.computer.vision;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import javax.imageio.ImageIO;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class NeuralNetworkForPictureEnchancement2 {

    public static void main(String[] args) {

        BufferedImage inputImage = loadImage("art1.jpg");

        List<double[][]> trainingDataList = prepareTrainingDataList("art2.jpg", "art3.jpg", "art4.jpg", "art5.jpg", "art6.jpg", "art7.jpg", "art8.jpg", "art9.jpg", "art10.jpg");

        NeuralNetwork neuralNetwork1 = new NeuralNetwork(1, 1, 8, .5);
        NeuralNetwork neuralNetwork2 = new NeuralNetwork(1, 1, 8, .5);
        NeuralNetwork neuralNetwork3 = new NeuralNetwork(1, 1, 8, .5);

        trainNeuralNetworksConcurrently(neuralNetwork1, neuralNetwork2, neuralNetwork3, trainingDataList);

        BufferedImage enhancedImage = enhanceImage(inputImage, neuralNetwork1, neuralNetwork2, neuralNetwork3);

        saveImage(enhancedImage, "enhanced.jpg");
    }

    public static BufferedImage loadImage(String imagePath) {
        System.out.println("Loading image: " + imagePath);
        try {
            return ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void saveImage(BufferedImage image, String outputPath) {
        try {
            ImageIO.write(image, "JPEG", new File(outputPath));
        } catch (IOException e) {
        }
    }

    public static List<double[][]> prepareTrainingDataList(String... targetImagePaths) {
        List<double[][]> trainingDataList = new ArrayList<>();

        for (String targetImagePath : targetImagePaths) {
            BufferedImage targetImage = loadImage(targetImagePath);
            int width = targetImage.getWidth();
            int height = targetImage.getHeight();

            double[][] trainingData = new double[width * height][2];
            int index = 0;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    trainingData[index][0] = (targetImage.getRGB(x, y) & 0xFF) / 255.0;
                    trainingData[index][1] = (targetImage.getRGB(x, y) & 0xFF) / 255.0;
                    index++;
                }
            }

            trainingDataList.add(trainingData);
        }

        return trainingDataList;
    }

    public static void trainNeuralNetworksConcurrently(NeuralNetwork neuralNetwork1, NeuralNetwork neuralNetwork2,
            NeuralNetwork neuralNetwork3, List<double[][]> trainingDataList) {
        int epochs = 20;
        double learningRate = 10;

        ExecutorService executorService = Executors.newFixedThreadPool(3);

        List<Future<Void>> futures = new ArrayList<>();

        futures.add(executorService.submit(() -> {
            trainNeuralNetwork(neuralNetwork1, trainingDataList.get(0), epochs, learningRate);
            return null;
        }));

        futures.add(executorService.submit(() -> {
            trainNeuralNetwork(neuralNetwork2, trainingDataList.get(1), epochs, learningRate);
            return null;
        }));

        futures.add(executorService.submit(() -> {
            trainNeuralNetwork(neuralNetwork3, trainingDataList.get(2), epochs, learningRate);
            return null;
        }));

        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
            }
        }

        executorService.shutdown();
    }

    public static void trainNeuralNetwork(NeuralNetwork neuralNetwork, double[][] trainingData, int epochs,
            double learningRate) {
        Random random = new Random();

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int i = trainingData.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                double[] temp = trainingData[i];
                trainingData[i] = trainingData[j];
                trainingData[j] = temp;
            }

            for (double[] data : trainingData) {
                double[] input = {data[0]};
                double[] target = {data[1]};
                neuralNetwork.train(input, target, learningRate);
            }

            double errorRate = calculateErrorRate(neuralNetwork, trainingData);
            System.out.println("Epoch: " + (epoch + 1) + ", Error Rate: " + errorRate);
        }
    }

    public static double calculateErrorRate(NeuralNetwork neuralNetwork, double[][] trainingData) {
        double totalError = 0.0;

        for (double[] data : trainingData) {
            double[] input = {data[0]};
            double[] target = {data[1]};
            double[] output = neuralNetwork.predict(input);

            double error = Math.abs(output[0] - target[0]);
            totalError += error;
        }

        return totalError / trainingData.length;
    }

    public static BufferedImage enhanceImage(BufferedImage inputImage, NeuralNetwork neuralNetwork1,
            NeuralNetwork neuralNetwork2, NeuralNetwork neuralNetwork3) {
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        BufferedImage enhancedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double[] input = {(inputImage.getRGB(x, y) & 0xFF) / 255.0};
                double prediction1 = neuralNetwork1.predict(input)[0];
                double prediction2 = neuralNetwork2.predict(input)[0];
                double prediction3 = neuralNetwork3.predict(input)[0];

                double enhancedValue = (prediction1 + prediction2 + prediction3) / 3.0;
                int enhancedPixel = (int) (enhancedValue * 255.0);

                Color enhancedColor = new Color(enhancedPixel, enhancedPixel, enhancedPixel);
                enhancedImage.setRGB(x, y, enhancedColor.getRGB());
            }
        }

        return enhancedImage;
    }

    public static class NeuralNetwork {

        private final int inputSize;
        private final int outputSize;
        private final int hiddenSize;
        private final double dropoutRate;
        private double[][] weights1;
        private double[][] weights2;

        public NeuralNetwork(int inputSize, int outputSize, int hiddenSize, double dropoutRate) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.hiddenSize = hiddenSize;
            this.dropoutRate = dropoutRate;

            this.weights1 = new double[inputSize][hiddenSize];
            this.weights2 = new double[hiddenSize][outputSize];

            Random random = new Random();

            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    weights1[i][j] = random.nextDouble();
                }
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights2[i][j] = random.nextDouble();
                }
            }
        }

        public double[] predict(double[] input) {
            double[] hidden = dotProduct(input, weights1);
            hidden = applySigmoid(hidden);

            double[] output = dotProduct(hidden, weights2);
            output = applySigmoid(output);

            return output;
        }

        public void train(double[] input, double[] target, double learningRate) {
            double[] hidden = dotProduct(input, weights1);
            hidden = applySigmoid(hidden);

            hidden = applyDropout(hidden);

            double[] output = dotProduct(hidden, weights2);
            output = applySigmoid(output);

            double[] outputError = subtract(target, output);
            double[] hiddenError = dotProduct(outputError, transpose(weights2));

            double[] outputDelta = multiply(outputError, applySigmoidDerivative(output));
            double[] hiddenDelta = multiply(hiddenError, applySigmoidDerivative(hidden));

            double[][] weights2Adjustment = dotProduct(transpose(new double[][]{hidden}), new double[][]{outputDelta});
            double[][] weights1Adjustment = dotProduct(transpose(new double[][]{input}), new double[][]{hiddenDelta});

            weights2 = subtract(weights2, multiply(weights2Adjustment, learningRate));
            weights1 = subtract(weights1, multiply(weights1Adjustment, learningRate));
        }

        private double[] applyDropout(double[] x) {
            if (dropoutRate == 0.0) {
                return x;
            }

            double[] mask = new double[x.length];
            Random random = new Random();

            for (int i = 0; i < x.length; i++) {
                if (random.nextDouble() >= dropoutRate) {
                    mask[i] = 1.0 / (1.0 - dropoutRate);
                }
            }

            return multiply(x, mask);
        }

        private double[] dotProduct(double[] a, double[][] b) {
            int m = a.length;
            int n = b[0].length;
            double[] c = new double[n];

            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    c[j] += a[i] * b[i][j];
                }
            }

            return c;
        }

        private double[][] dotProduct(double[][] a, double[][] b) {
            int m = a.length;
            int n = b[0].length;
            int p = a[0].length;
            double[][] c = new double[m][n];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < p; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }

            return c;
        }

        private double[] applySigmoid(double[] x) {
            double[] result = new double[x.length];

            for (int i = 0; i < x.length; i++) {
                result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
            }

            return result;
        }

        private double[][] applySigmoid(double[][] x) {
            double[][] result = new double[x.length][x[0].length];

            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++) {
                    result[i][j] = 1.0 / (1.0 + Math.exp(-x[i][j]));
                }
            }

            return result;
        }

        private double[] applySigmoidDerivative(double[] x) {
            double[] result = new double[x.length];

            for (int i = 0; i < x.length; i++) {
                double sigmoid = 1.0 / (1.0 + Math.exp(-x[i]));
                result[i] = sigmoid * (1 - sigmoid);
            }

            return result;
        }

        private double[][] transpose(double[][] matrix) {
            int m = matrix.length;
            int n = matrix[0].length;
            double[][] result = new double[n][m];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    result[i][j] = matrix[j][i];
                }
            }

            return result;
        }

        private double[] subtract(double[] a, double[] b) {
            int n = a.length;
            double[] result = new double[n];

            for (int i = 0; i < n; i++) {
                result[i] = a[i] - b[i];
            }

            return result;
        }

        private double[][] subtract(double[][] a, double[][] b) {
            int m = a.length;
            int n = a[0].length;
            double[][] result = new double[m][n];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] = a[i][j] - b[i][j];
                }
            }

            return result;
        }

        private double[] multiply(double[] a, double[] b) {
            int n = a.length;
            double[] result = new double[n];

            for (int i = 0; i < n; i++) {
                result[i] = a[i] * b[i];
            }

            return result;
        }

        private double[][] multiply(double[][] a, double b) {
            int m = a.length;
            int n = a[0].length;
            double[][] result = new double[m][n];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] = a[i][j] * b;
                }
            }

            return result;
        }

    }
}
