package ac.up.cos700.neutralitystudy.neuralnet.util;

import ac.up.cos700.neutralitystudy.function.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Configuration object used by the network builder to specify a network
 * topology. For now, all neurons in a layer are constrained to the same
 * activation function, since more granular specifications will probably not
 * be needed.
 * 
 * @author Abrie van Aardt
 */
public class FFNeuralNetConfig {
    public List<LayerConfig> layers = new ArrayList<>();
}
