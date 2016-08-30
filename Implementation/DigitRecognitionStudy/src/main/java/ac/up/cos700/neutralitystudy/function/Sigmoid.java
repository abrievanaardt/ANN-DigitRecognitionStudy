package ac.up.cos700.neutralitystudy.function;

import ac.up.cos700.neutralitystudy.function.IFunction;
import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;

/**
 * Implementation of Sigmoid in 1D with λ = 1
 *
 * @author Abrie van Aardt
 */
public class Sigmoid implements IFunction {

    double lambda;

    public Sigmoid() {
        lambda = 1;
    }
    
    public Sigmoid(double _lambda){
        lambda = _lambda;
    }

    @Override
    public int getDimensionality() {
        return 1;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 1)
            throw new UnequalArgsDimensionException();

        return 1.0 / (1.0 + Math.pow(Math.E, -lambda * x[0]));
    }

}
