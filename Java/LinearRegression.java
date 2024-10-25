import java.util.Random;

public class LinearRegression {
    private double learningRate;
    private int iterations;
    private double weight;
    private double bias;

    public LinearRegression(double learningRate, int iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.weight = 0;
        this.bias = 0;
    }

    // Method to train the linear regression model using gradient descent
    public void train(double[] X, double[] Y) {
        int n = X.length;

        for (int i = 0; i < iterations; i++) {
            double weightGradient = 0;
            double biasGradient = 0;

            // Calculate the gradients
            for (int j = 0; j < n; j++) {
                double prediction = predict(X[j]);
                weightGradient += (prediction - Y[j]) * X[j];
                biasGradient += (prediction - Y[j]);
            }

            // Update the weight and bias using the gradients
            weight -= learningRate * (weightGradient / n);
            bias -= learningRate * (biasGradient / n);
        }
    }

    // Method to make predictions using the learned weight and bias
    public double predict(double x) {
        return weight * x + bias;
    }

    // Method to evaluate the model on test data
    public double meanSquaredError(double[] X, double[] Y) {
        int n = X.length;
        double mse = 0;

        for (int i = 0; i < n; i++) {
            double prediction = predict(X[i]);
            mse += Math.pow(prediction - Y[i], 2);
        }

        return mse / n;
    }

    // Main method for testing the Linear Regression model
    public static void main(String[] args) {
        // Generate some random training data
        int dataSize = 100;
        double[] X = new double[dataSize];
        double[] Y = new double[dataSize];
        
        // Generate a simple linear relationship: Y = 2X + 3 + noise
        Random random = new Random();
        for (int i = 0; i < dataSize; i++) {
            X[i] = random.nextDouble() * 10;
            Y[i] = 2 * X[i] + 3 + random.nextGaussian(); // Adding Gaussian noise for randomness
        }

        // Create and train the model
        LinearRegression model = new LinearRegression(0.01, 1000);
        model.train(X, Y);

        // Test the model and calculate mean squared error on training data
        double mse = model.meanSquaredError(X, Y);
        System.out.println("Mean Squared Error: " + mse);

        // Make a prediction with the trained model
        double testValue = 5.0;
        double prediction = model.predict(testValue);
        System.out.println("Prediction for X = " + testValue + ": " + prediction);
    }
}
