package ac.up.cos711.digitrecognitionstudy.neuralnet.metric;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.IFFNeuralNet;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;

/**
 *
 * @author Abrie van Aardt
 */
public interface INetworkError {
    public double measure(IFFNeuralNet network, Dataset testingSet)
            throws UnequalInputWeightException, UnequalArgsDimensionException;
}
