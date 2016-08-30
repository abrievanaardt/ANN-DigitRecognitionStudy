package ac.up.cos700.neutralitystudy.function;

import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;

/**
 *
 * @author Abrie van Aardt
 */
public class NormalisedMeanSquaredError extends MeanSquaredError{
    @Override
    //TODO: normalise
    public double evaluate(double... x) throws UnequalArgsDimensionException{
        return super.evaluate(x);
    }
}
