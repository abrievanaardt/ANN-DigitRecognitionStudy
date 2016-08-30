package ac.up.cos711.digitrecognitionstudy.neuralnet.util;

/**
 *
 * @author Abrie van Aardt
 */
public class UnequalInputWeightException extends Exception{
    @Override
    public String getMessage(){
        return "Input vector must match weight vector in length";
    }
}
