package ac.up.cos711.digitrecognitionstudy.neuralnet;

import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;

/**
 * Interface to a fully connected feed forward neural network for
 * classification.
 *
 * @author Abrie van Aardt
 */
public interface IFFNeuralNet {

    /**
     * Computes class probabilities for the given input pattern (classifies the
     * data).
     *
     * @param inputPattern
     * @return array of class probabilities
     * @throws UnequalInputWeightException
     * @throws UnequalArgsDimensionException
     */
    public double[] classify(double... inputPattern)
            throws UnequalInputWeightException, UnequalArgsDimensionException;

    /**
     * Return an array (vector) of the weights in the network in order of layer
     * occurrence.
     *
     * @return weightVector
     */
    public double[] getWeightVector();

    /**
     * Treats the weights in the neural net as a vector, assigning them the
     * values in _weightVector in order of layer occurrence.
     *
     * @param _weightVector
     * @throws UnequalArgsDimensionException
     */
    public void setWeightVector(double... _weightVector)
            throws UnequalArgsDimensionException;

    /**
     * Gets the number of weights in the entire network, including weight
     * biases.
     *
     * @return network dimensionality
     */
    public int getDimensionality();

    /**
     * Acquires a live reference to the underlying neurons in the network. This
     * facilitates learning algorithms such as {@link BackPropogation} which
     * requires knowledge of the network topology.
     *
     * @return Zagged 2D array of Neurons
     */
    public Neuron[][] getNetworkLayers();

}
