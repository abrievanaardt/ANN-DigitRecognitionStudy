package ac.up.cos700.neutralitystudy.neuralnet.training;

import ac.up.cos700.neutralitystudy.data.Dataset;
import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;
import ac.up.cos700.neutralitystudy.neuralnet.IFFNeuralNet;
import ac.up.cos700.neutralitystudy.neuralnet.util.UnequalInputWeightException;

/**
 * Interface to the functionality to train a fully connected feed forward
 * neural network. Search/training algorithms intended to be used to search
 * weight space for minimal error must implement this interface.
 * 
 * @author Abrie van Aardt
 */
public interface IFFNeuralNetTrainer{
    /**
     * This method will adjust the network weights to produce minimal error
     * on the dataset passed as the second parameter. Care should be taken to 
     * ensure only the training set of a larger dataset is passed as a view by
     * calling
     * <pre>
     *  dataset.shuffle().split(ratio);
     * </pre> on a {@link Dataset} object in client code.
     * @param network
     * @param dataset
     * @throws UnequalInputWeightException
     * @throws UnequalArgsDimensionException 
     */
    public void train(IFFNeuralNet network, Dataset dataset)
            throws UnequalInputWeightException, UnequalArgsDimensionException;
}
