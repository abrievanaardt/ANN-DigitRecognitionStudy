package ac.up.cos700.neutralitystudy.neuralnet.util;

/**
 *
 * @author Abrie van Aardt
 */
public class ZeroNeuronException extends Exception {
    @Override
    public String getMessage(){
        return "There has to be at least one neuron in a layer";
    }
}
