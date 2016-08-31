package ac.up.cos711.digitrecognitionstudy.function;

import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.FFNeuralNet;

/**
 * This implements the Softmax activation function. Since global information 
 * is necessary to complete the computation, the {@link FFNeuralNet} is
 * responsible for computing and dividing by the sum of the output nodes.
 * 
 * @author Abrie van Aardt
 */
public class Softmax implements IFunction {

    @Override
    public int getDimensionality() {
        return 1;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 1)
            throw new UnequalArgsDimensionException();
        
        return Math.pow(Math.E,x[0]);
    }

}
