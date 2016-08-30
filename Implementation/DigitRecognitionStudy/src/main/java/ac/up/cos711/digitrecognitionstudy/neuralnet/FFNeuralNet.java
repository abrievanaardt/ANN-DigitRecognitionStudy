package ac.up.cos711.digitrecognitionstudy.neuralnet;

import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.FFNeuralNetConfig;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Default implementation of an {@link IFFNeuralNet}.
 *
 * @author Abrie van Aardt
 */
public class FFNeuralNet implements IFFNeuralNet {

    public FFNeuralNet(FFNeuralNetConfig config) {
        layers = new Neuron[config.layers.size()][];        
        
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Neuron[config.layers.get(i).neuronCount];
            for (int j = 0; j < layers[i].length; j++) {
                layers[i][j] = new Neuron();
                layers[i][j].setActivationFunction(config.layers.get(i).activationFunction);
                layers[i][j].setWeightCount(config.layers.get(i).weightCountPerNeuron);
            }
        }
    }

    @Override
    public double[] classify(double... inputPattern) throws UnequalInputWeightException, UnequalArgsDimensionException {
        double[] tempInputPattern;
        double[] tempOutputPattern;

        //input pattern dimension must match number of input neurons
        if (inputPattern.length != layers[0].length)
            throw new UnequalArgsDimensionException();

        //set output pattern length to number of neurons in this layer
        tempOutputPattern = new double[inputPattern.length];

        //feed inputPattern individually to neurons in the input layer
        for (int i = 0; i < inputPattern.length; i++) {
            tempOutputPattern[i] = layers[0][i].feed(inputPattern[i]);
        }

        //current output becomes the input for the next layer
        tempInputPattern = tempOutputPattern;

        for (int i = 1; i < layers.length; i++) {
            //set output pattern length to number of neurons in current layer
            tempOutputPattern = new double[layers[i].length];
            //feed input pattern to all neurons in current layer
            for (int j = 0; j < tempOutputPattern.length; j++) {
                tempOutputPattern[j] = layers[i][j].feed(tempInputPattern);
            }
            tempInputPattern = tempOutputPattern;
        }
        
        return tempOutputPattern;
    }

    @Override
    public int getDimensionality() {
        int dim = 0;
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                dim += layers[i][j].getWeightCount();
            }
        }
        return dim;
    }

    @Override
    public double[] getWeightVector() {
        double[] weightVector = new double[getDimensionality()];
        int weightVectorIndex = 0;
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                for (int k = 0; k < layers[i][j].getWeightCount(); k++) {
                    weightVector[weightVectorIndex++] = layers[i][j].getWeightAt(k);
                }
            }
        }
        return weightVector;
    }

    @Override
    public void setWeightVector(double... _weightVector) throws UnequalArgsDimensionException {
        if (_weightVector.length != getDimensionality())
            throw new UnequalArgsDimensionException();

        int weightVectorIndex = 0;
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                for (int k = 0; k < layers[i][j].getWeightCount(); k++) {
                    layers[i][j].setWeight(k, _weightVector[weightVectorIndex++]);
                }
            }
        }
    }

    @Override
    public Neuron[][] getNetworkLayers() {
        //handle with care, no pun intended
        return layers;
    }
    
    private Neuron[][] layers;  

    

}
