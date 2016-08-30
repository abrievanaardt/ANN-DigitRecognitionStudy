package ac.up.cos700.neutralitystudy.function;

import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;

/**
 * This serves as the services contract for all benchmark functions, including
 * statically defined functions such as {@link Alpine}, as well as functions with
 * configurable neutrality (e.g. stacking plates) and ultimately {@link MeanSquaredError}, for 
 * calculation of network error.
 * 
 * @author Abrie van Aardt 
 */
public interface IFunction {
    public int getDimensionality();
    public double evaluate(double... x) throws UnequalArgsDimensionException;        
}
