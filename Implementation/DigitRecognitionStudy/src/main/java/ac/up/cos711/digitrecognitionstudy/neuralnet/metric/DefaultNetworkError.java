package ac.up.cos711.digitrecognitionstudy.neuralnet.metric;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.Pattern;
import ac.up.cos711.digitrecognitionstudy.function.IFunction;
import ac.up.cos711.digitrecognitionstudy.function.LogLikelihood;
import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.IFFNeuralNet;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;
import java.util.Iterator;

/**
 * Network error that is used for Validation and Generalisation Tests. This
 * class implements the {@link LogLikelihood} function, with respect to
 * {@link SoftMax}, over the total patterns in the dataset.
 *
 * @author Abrie van Aardt
 */
public class DefaultNetworkError implements INetworkError {

    @Override
    public double measure(IFFNeuralNet network, Dataset testingSet)
            throws UnequalInputWeightException, UnequalArgsDimensionException {
        double error = 0;

        Iterator<Pattern> patterns = testingSet.iterator();
        double[] outputs;
        double[] targets;
        while (patterns.hasNext()) {

            Pattern p = patterns.next();
            outputs = network.classify(p.getInputs());
            targets = p.getTargets();
            error += errorForPattern(targets, outputs);      
        }

        error /= -testingSet.size();
        
        return error;
    }

    private double errorForPattern(double[] targets, double[] outputs) throws UnequalArgsDimensionException {
        double sum = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            sum += new LogLikelihood().evaluate(targets[i], outputs[i]);
        }
        return sum;
    }    

}
