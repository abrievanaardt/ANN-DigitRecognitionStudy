package ac.up.cos711.digitrecognitionstudy.function;

import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;

/**
 *
 * @author Abrie van Aardt
 */
public class SquaredError implements IFunction{

    @Override
    public int getDimensionality() {
        return 2;
    }

    /**
     * Expects the target first, followed by the output of which the difference
     * is then squared.
     * 
     * @param x the target and the output
     * @return squared difference
     * @throws UnequalArgsDimensionException 
     */
    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 2)
            throw new UnequalArgsDimensionException();
        
        return Math.pow(x[0] - x[1], 2);
    }

}
