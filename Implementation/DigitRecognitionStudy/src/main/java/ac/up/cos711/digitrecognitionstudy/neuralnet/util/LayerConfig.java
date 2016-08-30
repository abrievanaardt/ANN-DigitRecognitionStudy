package ac.up.cos711.digitrecognitionstudy.neuralnet.util;

import ac.up.cos711.digitrecognitionstudy.function.IFunction;
import ac.up.cos711.digitrecognitionstudy.function.Sigmoid;

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
