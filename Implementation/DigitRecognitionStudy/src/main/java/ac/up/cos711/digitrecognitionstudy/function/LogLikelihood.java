package ac.up.cos711.digitrecognitionstudy.function;

import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;

/**
 * Calculates the Log-Likelihood error of a single output node produced by a
 * single pattern.
 * 
 * @author Abrie van Aardt
 */
public class LogLikelihood implements IFunction{
    @Override
    public int getDimensionality() {
        return 2;
    }

    /**
     * Expects the target first, followed by the output at node k pattern p.
     * @param x the target and output values at node k pattern p
     * @return the log-likelihood error
     * @throws UnequalArgsDimensionException 
     */
    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 2)
            throw new UnequalArgsDimensionException();
        
        return x[0] * Math.log(x[1]);//Math.log corresponds to ln
    }
}
