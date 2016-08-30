package ac.up.cos700.neutralitystudy.function;

import ac.up.cos700.neutralitystudy.data.Dataset;
import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;
import ac.up.cos700.neutralitystudy.neuralnet.IFFNeuralNet;

/**
 * 
 * 
 * @author Abrie van Aardt
 */
public class SumSquaredError extends NetworkError {

    public SumSquaredError(IFFNeuralNet _network, Dataset _dataset) {
        super(_network, _dataset);
    }

    /**
     * 
     *
     * @param x the weight vector for the neural net. Weights are ordered
     * according to their layer indices.
     */
    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        return 0.0;
    }
}
