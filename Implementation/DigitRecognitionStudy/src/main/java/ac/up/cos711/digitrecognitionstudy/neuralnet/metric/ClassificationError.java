package ac.up.cos711.digitrecognitionstudy.neuralnet.metric;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.Pattern;
import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.IFFNeuralNet;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.ThresholdOutOfBoundsException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;
import java.util.Iterator;

/**
 * This class measures the % of incorrectly classified data patterns. The
 decision of whether a particular pattern belongs to any class is influenced
 by the THRESHOLD parameter. THRESHOLD should always be in the range [0, (Max_t
 - Min_t)/2] => [0,0.4] in the case of 0.1 and 0.9 targets.
 *
 * @author Abrie van Aardt
 */
public class ClassificationError implements INetworkError {

    public ClassificationError() {
        RIGOR = 0.2;//this is within bounds
    }

    public ClassificationError(double _threshold) throws ThresholdOutOfBoundsException {
        if (_threshold < 0 || _threshold > 0.4)
            throw new ThresholdOutOfBoundsException();
        RIGOR = _threshold;
    }

    /**
     * Calculates the % of patterns that were not correctly classified by the
     * network on the particular dataset.
     *
     * @param network
     * @param testingSet
     * @return % of patterns incorrectly classified
     * @throws UnequalInputWeightException
     * @throws UnequalArgsDimensionException
     */
    @Override
    public double measure(IFFNeuralNet network, Dataset testingSet)
            throws UnequalInputWeightException, UnequalArgsDimensionException {

        int correctClassCount = 0;

        Iterator<Pattern> testIter = testingSet.iterator();
        while (testIter.hasNext()) {
            Pattern p = testIter.next();
            double[] outputs = network.classify(p.getInputs());
            double[] targets = p.getTargets();
            int correctNodeCount = 0;
            
            for (int i = 0; i < outputs.length; i++) {
                if (isCorrect(targets[i], outputs[i]))
                    ++correctNodeCount;
            }
            
            if (correctNodeCount == outputs.length)
                ++correctClassCount;
        }
        
        double percentage = (testingSet.size() - correctClassCount) 
                / ((double) testingSet.size()) * 100.0;

        return percentage;
    }

    private boolean isCorrect(double target, double output) {
        return Math.abs(target - output) <= 0.4 - RIGOR;
    }

    private final double RIGOR;
}
