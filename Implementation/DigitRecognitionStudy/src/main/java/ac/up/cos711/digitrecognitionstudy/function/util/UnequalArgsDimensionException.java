package ac.up.cos711.digitrecognitionstudy.function.util;

/**
 *
 * @author Abrie van Aardt
 */
public class UnequalArgsDimensionException extends Exception{
    @Override
    public String getMessage(){
        return "Number of input arguments must match the dimensionality of the function";
    }
}
