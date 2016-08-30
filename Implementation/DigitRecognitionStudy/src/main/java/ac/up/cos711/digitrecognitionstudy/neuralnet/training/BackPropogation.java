package ac.up.cos711.digitrecognitionstudy.neuralnet.training;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.Pattern;
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
 * function for hidden and output nodes. It is important that the dataset be
 * normalized for the active domain of Sigmoid. Both inputs and outputs alike.
 * Total network error is normalized over the output nodes and patterns in the
 * dataset to a value in the range [0,1).
 *
 * @author Abrie van Aardt
 */
public class BackPropogation implements IFFNeuralNetTrainer {

    
    //TODO: PLEASE MAKE USE OF ONE OF THE STANDARD ERROR FUNCTIONS!!!!
    //TODO: maybe make use of an x-Squared Error function to decide when to stop
    public BackPropogation() {
        errorDelta = 0.1;//todo: find good value
        learningRate = 0.01;//todo: find good value
    }

    public BackPropogation(double _errorDelta, double _learningRate) {
        errorDelta = _errorDelta;
        learningRate = _learningRate;
    }

    @Override
    public void train(IFFNeuralNet network, Dataset dataset)
            throws UnequalInputWeightException, UnequalArgsDimensionException {

        Logger.getLogger(getClass().getName())
                .log(Level.INFO, "Started neural network training...");

        initialise(network);
        double networkError;
        double[] outputs;
        double[] targets;
        double[] errors;
        int epoch = 0;
        long duration = System.nanoTime();

        do {
            //prevent memorisation of pattern order
            dataset.shuffle();
            Iterator<Pattern> patterns = dataset.iterator();
            networkError = 0;

            while (patterns.hasNext()) {

                Pattern p = patterns.next();
                outputs = network.classify(p.getInputs());

                //calculate the error for each output node
                errors = new double[outputs.length];
                targets = p.getTargets();
                for (int i = 0; i < outputs.length; i++) {
                    errors[i] = targets[i] - outputs[i];
                    //update total network error (implies accuracy)   
                    networkError += Math.abs(errors[i]) / (0.9 * errors.length);
                }

                backPropogateError(network, errors, outputs);
            }

            //normalize the network error
            networkError /= dataset.size();//range [0,1)   
            
            ++epoch;
        }
        while (networkError > errorDelta);
        
        duration = System.nanoTime() - duration;

        Logger.getLogger(getClass().getName())
                .log(Level.INFO, "Training completed in {0} epoch(s) ({1}s) with "
                        + "acceptable classification error of {2}.",
                        new Object[]{
                            epoch,
                            duration/1000000000,
                            networkError
                        }
                );
    }

    private void backPropogateError(IFFNeuralNet network, double[] errors, double[] outputs) {
        //obtain neurons
        Neuron[][] layers = network.getNetworkLayers();

        //calculate error signals from output nodes
        double[] errorSignals = new double[errors.length];
        for (int i = 0; i < errors.length; i++) {
            errorSignals[i] = -errors[i] * (1 - outputs[i]) * outputs[i];
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
                    updateWeight(layers, errorSignals, i, j, k, WeightType.NORMAL);
                    updateErrorSignal(layers, newErrorSignals, errorSignals, i, j, k);
                }
                //now adjust the bias weight
                biasIndex = layers[i][j].getWeightCount() - 1;
                updateWeight(layers, errorSignals, i, j, biasIndex, WeightType.BIAS);
            }

            //update error signals to be used for the next layer
            errorSignals = newErrorSignals;
        }

    }

    private void updateWeight(Neuron[][] layers, double[] errorSignals, int i, int j, int k, WeightType type) {
        double oldWeight;
        double newWeight;
        oldWeight = layers[i][j].getWeightAt(k);
        newWeight = -learningRate * errorSignals[j];

        if (type == WeightType.NORMAL) {
            newWeight *= layers[i - 1][k].getOutput();//input for weight_k
        }
        else if (type == WeightType.BIAS) {
            newWeight *= -1;//input = -1 for bias            
        }

        newWeight += oldWeight;

        layers[i][j].setWeight(k, newWeight);
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
    private double errorDelta;
    private double learningRate;

    private enum WeightType {
        BIAS, NORMAL
    };

}
