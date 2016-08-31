package ac.up.cos711.digitrecognitionstudy.neuralnet.training;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.Pattern;
import ac.up.cos711.digitrecognitionstudy.function.LogLikelihood;
import ac.up.cos711.digitrecognitionstudy.function.Softmax;
import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.IFFNeuralNet;
import ac.up.cos711.digitrecognitionstudy.neuralnet.Neuron;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;
import java.util.Iterator;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements the BackPropogation algorithm, assuming the Sigmoid activation
 * function for hidden and Sigmoid/Softmax for output nodes. It is important
 * that the dataset be normalized for the active domain of Sigmoid. Total
 * network error is normalized over the output nodes and patterns in the dataset
 * to a value in the range [0,1).
 *
 * @author Abrie van Aardt
 */
public class BackPropogation implements IFFNeuralNetTrainer {

    public BackPropogation() {
        ERROR_DELTA = 0.1;//todo: find good value
        LEARNING_RATE = 0.01;//todo: find good value
        BIN_SIZE = 10;//10 for MNIST data
    }

    public BackPropogation(double _errorDelta, double _learningRate, int _binSize) {
        ERROR_DELTA = _errorDelta;
        LEARNING_RATE = _learningRate;
        BIN_SIZE = _binSize;
    }

    @Override
    public void train(IFFNeuralNet network, Dataset dataset)
            throws UnequalInputWeightException, UnequalArgsDimensionException {

        Logger.getLogger(getClass().getName())
                .log(Level.INFO, "Started neural network training...");

        initialise(network);
        double trainingError;
        double trainingErrorMSE;
        int patternNumber;
        double[] outputs;
        double[] targets;
        int epoch = 0;
        long duration = System.nanoTime();

        do {
            //prevent memorisation of pattern order
            dataset.shuffle();
            Iterator<Pattern> patterns = dataset.iterator();
            trainingError = 0;
            trainingErrorMSE = 0;
            patternNumber = 1;

            while (patterns.hasNext()) {
                int test = 0;
                if (patternNumber % 1000 == 0)
                    test++;
                int y = test;
                Pattern p = patterns.next();
                outputs = network.classify(p.getInputs());
                targets = p.getTargets();
                trainingError += errorForPattern(targets, outputs) / -(dataset.size() * outputs.length);
                backPropogateError(network, targets, outputs);
//                if (patternNumber % BIN_SIZE == 0)
//                    triggerWeightUpdates(network);
                ++patternNumber;
            }

            System.out.println(trainingError);

            //if last few patterns did not fill a bin, trigger a weight update
//            if (dataset.size() % BIN_SIZE != 0)
//                triggerWeightUpdates(network);
            ++epoch;
        }
        while (trainingError > ERROR_DELTA);

        duration = System.nanoTime() - duration;

        Logger.getLogger(getClass().getName())
                .log(Level.INFO, "Training completed in {0} epoch(s) ({1}s) with "
                        + "acceptable E_t of {2}.",
                        new Object[]{
                            epoch,
                            duration / 1000000000,
                            trainingError
                        }
                );
    }

    private double errorForPattern(double[] targets, double[] outputs) throws UnequalArgsDimensionException {
        double sum = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            sum += new LogLikelihood().evaluate(targets[i], outputs[i]);
        }
        return sum;
    }

    private void backPropogateError(IFFNeuralNet network, double[] targets, double[] outputs) {
        //obtain neurons
        Neuron[][] layers = network.getNetworkLayers();

        double[] errorSignals = new double[outputs.length];

        //calculate error signals from output nodes
        if (layers[layers.length - 1][0].getActivationFunction() == Softmax.class) {
            for (int i = 0; i < outputs.length; i++) {
                errorSignals[i] = -(outputs[i] - targets[i]);//todo: check correctness of derivative
            }
        }
        else {
            for (int i = 0; i < outputs.length; i++) {
                errorSignals[i] = -(targets[i] - outputs[i]) * (1 - outputs[i]) * outputs[i];
            }
        }

        int biasIndex;
        double[] newErrorSignals;

        //iterate through layers, from last to second to update weights
        //input layer is excluded since identity function is assumed
        for (int i = layers.length - 1; i >= 1; i--) {
            //used to capture error signals for the next layer
            newErrorSignals = new double[layers[i - 1].length];

            for (int j = 0; j < layers[i].length; j++) {
                //adjust all weights excluding the bias
                for (int k = 0; k < layers[i][j].getWeightCount() - 1; k++) {
                    accumulateWeightDelta(layers, errorSignals, i, j, k, WeightType.NORMAL);
                    updateErrorSignal(layers, newErrorSignals, errorSignals, i, j, k);
                }
                //now adjust the bias weight
                biasIndex = layers[i][j].getWeightCount() - 1;
                accumulateWeightDelta(layers, errorSignals, i, j, biasIndex, WeightType.BIAS);
            }

            //update error signals to be used for the next layer
            errorSignals = newErrorSignals;
        }

    }

    private void triggerWeightUpdates(IFFNeuralNet network) {
        //obtain neurons
        Neuron[][] layers = network.getNetworkLayers();

        //update each weight in the network with its corresponding delta
        //that was accumulated over BIN_SIZE times
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                for (int k = 0; k < layers[i][j].getWeightCount(); k++) {
                    layers[i][j].setWeight(k,
                            layers[i][j].getWeightAt(k)
                            + layers[i][j].getWeightDeltaAt(k));
                    //reset weight delta for future use
                    layers[i][j].setWeightDelta(k, 0);
                }
            }
        }
    }

    private void accumulateWeightDelta(Neuron[][] layers, double[] errorSignals, int i, int j, int k, WeightType type) {
        double oldWeightDelta;
        double newWeightDelta;
        oldWeightDelta = layers[i][j].getWeightAt(k);
        newWeightDelta = -LEARNING_RATE * errorSignals[j];

        if (type == WeightType.NORMAL) {
            newWeightDelta *= layers[i - 1][k].getOutput();//input for weight_k
        }
        else if (type == WeightType.BIAS) {
            newWeightDelta *= -1;//input = -1 for bias            
        }

        newWeightDelta += oldWeightDelta;

        layers[i][j].setWeight(k, newWeightDelta);
    }

    private void updateErrorSignal(Neuron[][] layers, double[] newErrorSignals, double[] errorSignals, int i, int j, int k) {
        newErrorSignals[k] += layers[i][j].getWeightAt(k)
                * errorSignals[j]
                * (1 - layers[i - 1][k].getOutput())
                * layers[i - 1][k].getOutput();
    }

    /**
     * Initialize all weights to a value in the range
     * <pre>
     *  [-1/sqrt(fanin), 1/sqrt(fanin)]
     * </pre> where fanin = # weights leading to the neuron.
     *
     * @param network
     */
    private void initialise(IFFNeuralNet network) {
        Neuron[][] layers = network.getNetworkLayers();
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                int fanin = layers[i][j].getWeightCount();
                double range = 1.0 / Math.sqrt(fanin);
                for (int k = 0; k < fanin; k++) {
                    layers[i][j].setWeight(k, rand.nextDouble() * 2 * range - range);
                }
            }
        }
    }

    private Random rand = new Random(System.nanoTime());
    private final double ERROR_DELTA;
    private final double LEARNING_RATE;
    private final int BIN_SIZE;

    private enum WeightType {
        BIAS, NORMAL
    };

}
