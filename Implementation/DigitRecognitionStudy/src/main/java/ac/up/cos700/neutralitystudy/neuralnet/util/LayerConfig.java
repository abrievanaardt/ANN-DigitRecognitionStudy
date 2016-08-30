package ac.up.cos700.neutralitystudy.neuralnet.util;

import ac.up.cos700.neutralitystudy.function.IFunction;
import ac.up.cos700.neutralitystudy.function.Sigmoid;

/**
 * Encapsulates the information needed to build a single layer of the neural
 * network.
 * 
 * @author Abrie van Aardt
 */
public class LayerConfig {
    public IFunction activationFunction = new Sigmoid();
    public int weightCountPerNeuron = 0;
    public int neuronCount = 1;
}
