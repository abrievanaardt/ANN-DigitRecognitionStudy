package ac.up.cos700.neutralitystudy.function;

import ac.up.cos700.neutralitystudy.data.Dataset;
import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;
import ac.up.cos700.neutralitystudy.neuralnet.IFFNeuralNet;

/**
 * This class provides functionality to evaluate neural network classification
 * error given a weight vector. The network and accompanying dataset is
 * injected.
 *
 * @author Abrie van Aardt
 */
public abstract class NetworkError implements IFunction{
    private IFFNeuralNet network;
    private Dataset dataset;
    
    //todo: add dataset as parameter here
    public NetworkError(IFFNeuralNet _network, Dataset _dataset) {
        network = _network;
        dataset = _dataset;
    }
    
    /**
     * Calculates the number of weights that occur in the network, including 
     * the biases.
     * 
     * @return network weight dimensionality
     */
    @Override
    public int getDimensionality() {
        return network.getDimensionality();
    }

    /**
     * Evaluates the classification error of the network given all the weights
     * that should be used in the network, together with the dataset.
     * This method is intended to be overridden by specialized network error 
     * functions such as {@link SumSquaredError}.
     * 
     * @param x
     * @return network error 
     * @throws UnequalArgsDimensionException 
     */
    @Override
    public abstract double evaluate(double... x) throws UnequalArgsDimensionException ;
    
    
}
