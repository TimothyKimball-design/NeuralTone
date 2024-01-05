
package com.mycompany.computer.vision;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

public class NeuralNetworkForPictureEnhancement3 {

    public static void main(String[] args) {
        BufferedImage inputImage = loadImage("art1.jpg");
        List<double[][]> trainingDataList = prepareTrainingDataList(
                "art2.jpg", "art3.jpg", "art4.jpg", "art5.jpg", "art6.jpg", "art7.jpg", "art8.jpg", "art9.jpg", "art10.jpg", "art11.jpg", "art12.jpg", "art13.jpg", "art14.jpg", "art15.jpg", "art16.jpg", "art17.jpg", "art19.jpg", "art20.jpg", "art21.jpg", "art22.jpg", "art23.jpg", "art24.jpg", "art25.jpg", "art26.jpg", "art27.jpg", "art28.jpg", "art29.jpg", "art30.jpg", "art31.jpg", "art32.jpg", "art33.jpg", "art34.jpg", "art35.jpg", "art36.jpg", "art37.jpg");

        NeuralNetwork neuralNetwork1 = new NeuralNetwork(1, 1, 8, .2, 0.001, 0.001);
        NeuralNetwork neuralNetwork2 = new NeuralNetwork(1, 1, 8, .2, 0.001, 0.001);
        NeuralNetwork neuralNetwork3 = new NeuralNetwork(1, 1, 8, .2, 0.001, 0.001);

        trainNeuralNetworksConcurrently(neuralNetwork1, neuralNetwork2, neuralNetwork3, trainingDataList);

        Ensemble ensemble = new Ensemble(neuralNetwork1, neuralNetwork2, neuralNetwork3);

        BufferedImage enhancedImage = ensemble.enhanceImage(inputImage);
        saveImage(enhancedImage, "enhanced.png");

        displayImages(inputImage, enhancedImage);
    }

    public static void displayImages(BufferedImage originalImage, BufferedImage enhancedImage) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Image Viewer");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JPanel panel = new JPanel() {
                @Override
                protected void paintComponent(Graphics g) {
                    super.paintComponent(g);
                    g.drawImage(originalImage, 0, 0, getWidth() / 2, getHeight(), null);
                    g.drawImage(enhancedImage, getWidth() / 2, 0, getWidth() / 2, getHeight(), null);
                }
            };

            frame.getContentPane().add(panel);
            frame.setSize(800, 400);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
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
            e.printStackTrace();
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
                    int rgb = targetImage.getRGB(x, y);
                    trainingData[index][0] = ((rgb >> 16) & 0xFF) / 255.0;
                    trainingData[index][1] = ((rgb >> 8) & 0xFF) / 255.0;
                    index++;
                }
            }

          
            normalize(trainingData);

            trainingDataList.add(trainingData);
        }

        return trainingDataList;
    }

    private static void normalize(double[][] data) {

        double[] mean = new double[data[0].length];
        double[] stdDev = new double[data[0].length];

        for (int i = 0; i < data[0].length; i++) {
            double sum = 0.0;
            for (double[] row : data) {
                sum += row[i];
            }
            mean[i] = sum / data.length;

            double sumSquaredDiff = 0.0;
            for (double[] row : data) {
                sumSquaredDiff += Math.pow(row[i] - mean[i], 2);
            }
            stdDev[i] = Math.sqrt(sumSquaredDiff / data.length);
        }

        for (double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                row[i] = (row[i] - mean[i]) / stdDev[i];
            }
        }
    }

    public static void trainNeuralNetworksConcurrently(NeuralNetwork neuralNetwork1, NeuralNetwork neuralNetwork2,
            NeuralNetwork neuralNetwork3, List<double[][]> trainingDataList) {
        int epochs = 5;
        double learningRate = 10;

        ExecutorService executorService = Executors.newFixedThreadPool(3);
        List<Future<Void>> futures = new ArrayList<>();

        for (int i = 0; i < 3; i++) {
            int finalI = i;
            futures.add(executorService.submit(() -> {
                trainNeuralNetwork(getNeuralNetwork(finalI, neuralNetwork1, neuralNetwork2, neuralNetwork3),
                        trainingDataList.get(finalI), epochs, learningRate, 5);
                return null;
            }));
        }

        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        executorService.shutdown();
    }

    private static NeuralNetwork getNeuralNetwork(int i, NeuralNetwork neuralNetwork1, NeuralNetwork neuralNetwork2,
            NeuralNetwork neuralNetwork3) {
        switch (i) {
            case 0:
                return neuralNetwork1;
            case 1:
                return neuralNetwork2;
            case 2:
                return neuralNetwork3;
            default:
                throw new IllegalArgumentException("Invalid index: " + i);
        }
    }

    public static void trainNeuralNetwork(NeuralNetwork neuralNetwork, double[][] trainingData, int epochs, double initialLearningRate, double clipThreshold) {
        Random random = new Random();
        double learningRate = initialLearningRate;

        for (int epoch = 0; epoch < epochs; epoch++) {
            List<double[]> trainingDataList = Arrays.asList(trainingData);
            Collections.shuffle(trainingDataList, random);
            trainingDataList.toArray(trainingData);

            for (double[] data : trainingData) {
                double[] input = {data[0]};
                double[] target = {data[1]};
                neuralNetwork.train(input, target, learningRate, clipThreshold);
            }

            double errorRate = calculateErrorRate(neuralNetwork, trainingData);
            System.out.println("Epoch: " + (epoch + 1) + ", Error Rate: " + errorRate + ", Learning Rate: " + learningRate);

            learningRate = adjustLearningRate(initialLearningRate, epoch, 1);
        }
    }

    private static double adjustLearningRate(double initialLearningRate, int epoch, double errorRate) {
        
        double decayFactor = 0.9;

        
        if (epoch % 1 == 0) {
          
            if (errorRate > 0.5) {
                return initialLearningRate * decayFactor;
            }
        }

        return initialLearningRate;
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

    public static class Ensemble {

        private List<NeuralNetwork> neuralNetworks;

        public Ensemble(NeuralNetwork... networks) {
            this.neuralNetworks = new ArrayList<>(Arrays.asList(networks));
        }

        public BufferedImage enhanceImage(BufferedImage inputImage) {
            int width = inputImage.getWidth();
            int height = inputImage.getHeight();

            BufferedImage enhancedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double[] input = {((inputImage.getRGB(x, y) >> 16) & 0xFF) / 255.0};

                    double ensemblePrediction = 0.0;
                    for (NeuralNetwork neuralNetwork : neuralNetworks) {
                        ensemblePrediction += neuralNetwork.predict(input)[0];
                    }
                    ensemblePrediction /= neuralNetworks.size();

                    int enhancedPixel = (int) (ensemblePrediction * 255.0);
                    Color enhancedColor = new Color(enhancedPixel, enhancedPixel, enhancedPixel);
                    enhancedImage.setRGB(x, y, enhancedColor.getRGB());
                }
            }

            return enhancedImage;
        }
    }

    public static class NeuralNetwork {

        private final int inputSize;
        private final int outputSize;
        private final int hiddenSize;
        private final double dropoutRate;
        private final double l2RegularizationCoefficient1;
        private final double l2RegularizationCoefficient2;

        private double[][] weights1;
        private double[][] weights2;
        private double[][] weights3;
        private double[][] weights4;
        private double[][] weights5;
        private double[][] weights6;
        private double[][] weights7;
        private double[][] weights8;

        public NeuralNetwork(int inputSize, int outputSize, int hiddenSize, double dropoutRate,
                double l2RegularizationCoefficient1, double l2RegularizationCoefficient2) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.hiddenSize = hiddenSize;
            this.dropoutRate = dropoutRate;
            this.l2RegularizationCoefficient1 = l2RegularizationCoefficient1;
            this.l2RegularizationCoefficient2 = l2RegularizationCoefficient2;

            this.weights1 = initializeWeights(inputSize, hiddenSize);
            this.weights2 = initializeWeights(hiddenSize, hiddenSize);
            this.weights3 = initializeWeights(hiddenSize, hiddenSize);
            this.weights4 = initializeWeights(hiddenSize, hiddenSize);
            this.weights5 = initializeWeights(hiddenSize, hiddenSize);
            this.weights6 = initializeWeights(hiddenSize, hiddenSize);
            this.weights7 = initializeWeights(hiddenSize, hiddenSize);
            this.weights8 = initializeWeights(hiddenSize, outputSize);
        }

        private double[][] initializeWeights(int rows, int cols) {
            double[][] weights = new double[rows][cols];
            Random random = new Random();
            double scalingFactor = Math.sqrt(2.0 / rows);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    weights[i][j] = random.nextGaussian() * scalingFactor;
                }
            }

            return weights;
        }

        public double[] predict(double[] input) {
            double[] hidden1 = dotProduct(input, weights1);
            hidden1 = applyLeakyReLU(hidden1);

            double[] hidden2 = dotProduct(hidden1, weights2);
            hidden2 = applyLeakyReLU(hidden2);

            double[] hidden3 = dotProduct(hidden2, weights3);
            hidden3 = applyLeakyReLU(hidden3);

            double[] hidden4 = dotProduct(hidden3, weights4);
            hidden4 = applyLeakyReLU(hidden4);

            double[] hidden5 = dotProduct(hidden4, weights5);
            hidden5 = applyLeakyReLU(hidden5);

            double[] hidden6 = dotProduct(hidden5, weights6);
            hidden4 = applyLeakyReLU(hidden4);

            double[] hidden7 = dotProduct(hidden6, weights7);
            hidden5 = applyLeakyReLU(hidden5);

            double[] output = dotProduct(hidden7, weights8);
            output = applyLeakyReLU(output);

            return output;
        }

        private double calculateGradientNorm(double[][] gradients) {
            double sum = 0.0;
            for (double[] row : gradients) {
                for (double value : row) {
                    sum += Math.pow(value, 2);
                }
            }
            return Math.sqrt(sum);
        }

        public void train(double[] input, double[] target, double learningRate, double clipThreshold) {
            double[] hidden1 = dotProduct(input, weights1);
            hidden1 = applyLeakyReLU(hidden1);

            double[] hidden2 = dotProduct(hidden1, weights2);
            hidden2 = applyLeakyReLU(hidden2);

            double[] hidden3 = dotProduct(hidden2, weights3);
            hidden3 = applyLeakyReLU(hidden3);

            double[] hidden4 = dotProduct(hidden3, weights4);
            hidden4 = applyLeakyReLU(hidden4);

            double[] hidden5 = dotProduct(hidden4, weights5);
            hidden5 = applyLeakyReLU(hidden5);

            double[] hidden6 = dotProduct(hidden5, weights6);
            hidden4 = applyLeakyReLU(hidden4);

            double[] hidden7 = dotProduct(hidden6, weights7);
            hidden5 = applyLeakyReLU(hidden5);

            hidden1 = applyDropout(hidden1);
            hidden2 = applyDropout(hidden2);
            hidden3 = applyDropout(hidden3);
            hidden4 = applyDropout(hidden4);
            hidden5 = applyDropout(hidden5);
            hidden6 = applyDropout(hidden6);
            hidden7 = applyDropout(hidden7);

            double[] output = dotProduct(hidden7, weights8);
            output = applyLeakyReLU(output);

            double[] outputError = subtract(target, output);
            double[] hidden7Error = dotProduct(outputError, transpose(weights8));
            double[] hidden6Error = dotProduct(hidden7Error, transpose(weights7));
            double[] hidden5Error = dotProduct(hidden6Error, transpose(weights6));
            double[] hidden4Error = dotProduct(hidden5Error, transpose(weights5));
            double[] hidden3Error = dotProduct(hidden4Error, transpose(weights4));
            double[] hidden2Error = dotProduct(hidden3Error, transpose(weights3));
            double[] hidden1Error = dotProduct(hidden2Error, transpose(weights2));

            double[] outputDelta = multiply(outputError, applyLeakyReLUDerivative(output));
            double[] hidden7Delta = multiply(hidden7Error, applyLeakyReLUDerivative(hidden7));
            double[] hidden6Delta = multiply(hidden6Error, applyLeakyReLUDerivative(hidden6));
            double[] hidden5Delta = multiply(hidden5Error, applyLeakyReLUDerivative(hidden5));
            double[] hidden4Delta = multiply(hidden4Error, applyLeakyReLUDerivative(hidden4));
            double[] hidden3Delta = multiply(hidden3Error, applyLeakyReLUDerivative(hidden3));
            double[] hidden2Delta = multiply(hidden2Error, applyLeakyReLUDerivative(hidden2));
            double[] hidden1Delta = multiply(hidden1Error, applyLeakyReLUDerivative(hidden1));

            double[][] weights8Regularization = multiply(weights8, l2RegularizationCoefficient2);
            double[][] weights7Regularization = multiply(weights7, l2RegularizationCoefficient2);
            double[][] weights6Regularization = multiply(weights6, l2RegularizationCoefficient2);
            double[][] weights5Regularization = multiply(weights5, l2RegularizationCoefficient2);
            double[][] weights4Regularization = multiply(weights4, l2RegularizationCoefficient2);
            double[][] weights3Regularization = multiply(weights3, l2RegularizationCoefficient2);
            double[][] weights2Regularization = multiply(weights2, l2RegularizationCoefficient1);
            double[][] weights1Regularization = multiply(weights1, l2RegularizationCoefficient1);

            double[][] weights8Adjustment = dotProduct(transpose(new double[][]{hidden7}), new double[][]{outputDelta});
            double[][] weights7Adjustment = dotProduct(transpose(new double[][]{hidden6}), new double[][]{hidden7Delta});
            double[][] weights6Adjustment = dotProduct(transpose(new double[][]{hidden5}), new double[][]{hidden6Delta});
            double[][] weights5Adjustment = dotProduct(transpose(new double[][]{hidden4}), new double[][]{hidden5Delta});
            double[][] weights4Adjustment = dotProduct(transpose(new double[][]{hidden3}), new double[][]{hidden4Delta});
            double[][] weights3Adjustment = dotProduct(transpose(new double[][]{hidden2}), new double[][]{hidden3Delta});
            double[][] weights2Adjustment = dotProduct(transpose(new double[][]{hidden1}), new double[][]{hidden2Delta});
            double[][] weights1Adjustment = dotProduct(transpose(new double[][]{input}), new double[][]{hidden1Delta});

            double weights8GradientNorm = calculateGradientNorm(weights8Adjustment);
            if (weights8GradientNorm > clipThreshold) {
                double scalingFactor = clipThreshold / weights8GradientNorm;
                weights8Adjustment = multiply(weights8Adjustment, scalingFactor);
            }

            weights8 = subtract(weights8, multiply(weights8Adjustment, learningRate));
            weights7 = subtract(weights7, multiply(weights7Adjustment, learningRate));
            weights6 = subtract(weights6, multiply(weights6Adjustment, learningRate));
            weights5 = subtract(weights5, multiply(weights5Adjustment, learningRate));
            weights4 = subtract(weights4, multiply(weights4Adjustment, learningRate));
            weights3 = subtract(weights3, multiply(weights3Adjustment, learningRate));
            weights2 = subtract(weights2, multiply(weights2Adjustment, learningRate));
            weights1 = subtract(weights1, multiply(weights1Adjustment, learningRate));

            weights8 = subtract(weights8, weights8Regularization);
            weights7 = subtract(weights7, weights7Regularization);
            weights6 = subtract(weights6, weights6Regularization);
            weights5 = subtract(weights5, weights5Regularization);
            weights4 = subtract(weights4, weights4Regularization);
            weights3 = subtract(weights3, weights3Regularization);
            weights2 = subtract(weights2, weights2Regularization);
            weights1 = subtract(weights1, weights1Regularization);

            weights8 = subtract(weights8, multiply(weights8Adjustment, learningRate));
            weights7 = subtract(weights7, multiply(weights7Adjustment, learningRate));
            weights6 = subtract(weights6, multiply(weights6Adjustment, learningRate));
            weights5 = subtract(weights5, multiply(weights5Adjustment, learningRate));
            weights4 = subtract(weights4, multiply(weights4Adjustment, learningRate));
            weights3 = subtract(weights3, multiply(weights3Adjustment, learningRate));
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

        private double[] applyLeakyReLU(double[] x) {
            double[] result = new double[x.length];
            double alpha = 0.01;

            for (int i = 0; i < x.length; i++) {
                double value = x[i] > 0 ? x[i] : alpha * x[i];
                result[i] = Double.isFinite(value) ? value : 0.0;
            }

            return result;
        }

        private double[][] applyLeakyReLU(double[][] x) {
            double[][] result = new double[x.length][x[0].length];
            double alpha = 0.01;

            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++) {
                    double value = x[i][j] > 0 ? x[i][j] : alpha * x[i][j];
                    result[i][j] = Double.isFinite(value) ? value : 0.0;
                }
            }

            return result;
        }

        private double[] applyLeakyReLUDerivative(double[] x) {
            double[] result = new double[x.length];
            double alpha = 0.01;

            for (int i = 0; i < x.length; i++) {
                double derivative = x[i] > 0 ? 1 : alpha;
                result[i] = Double.isFinite(derivative) ? derivative : 0.0;
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
