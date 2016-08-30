package ac.up.cos700.neutralitystudy.function;

import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;

/**
 * Implementation of the identity function y = x
 *
 * @author Abrie van Aardt
 */
public class Identity implements IFunction {

    @Override
    public int getDimensionality() {
        return 1;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 1)
            throw new UnequalArgsDimensionException();
        return x[0];
    }
}
