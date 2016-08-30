package ac.up.cos700.neutralitystudy.neuralnet;

import ac.up.cos700.neutralitystudy.function.*;
import ac.up.cos700.neutralitystudy.function.util.*;
import ac.up.cos700.neutralitystudy.neuralnet.util.*;

/**
 * This is an implementation of a neural network unit, or neuron. By default
 * such a unit will make use of the Sigmoid activation function. The bias is
 * included in the weightVector as the last element.
 *
 * @author Abrie van Aardt
 */
public class Neuron {

    public Neuron() {
        weightVector = new double[0];
        activationFunction = new Sigmoid();
    }

    public Neuron(IFunction _activationFunction) {
        this();
        activationFunction = _activationFunction;
    }

    public void setActivationFunction(IFunction function) {
        activationFunction = function;
    }

    public void setWeightCount(int count) {
        weightVector = new double[count];
    }

    public int getWeightCount() {
        return weightVector.length;
    }

    public double getWeightAt(int index) {
        return weightVector[index];
    }

    public void setWeight(int index, double value) {
        weightVector[index] = value;
    }
    
    public double getOutput(){
        return output;
    }
    
    /**
     * Augments the input signal using a weight vector. Then calculates an
     * output signal using an activation function. 
     * 
     * @param inputVector
     * @return Output signal
     * @throws UnequalInputWeightException
     * @throws UnequalArgsDimensionException
     */
    public double feed(double... inputVector) throws UnequalInputWeightException, UnequalArgsDimensionException {
        if (inputVector.length != weightVector.length - 1 &&
                (inputVector.length != 1 || weightVector.length != 0)) {
            throw new UnequalInputWeightException();
        }
        
        output = activationFunction.evaluate(aggregate(inputVector));
        return output;
    }

    /**
     * Calculates the net input to the neuron as the weighted sum of the input
     * signal. This is not implemented as an {@link IFunction} since the strict
     * requirement of the interface will impair performance (weights and inputs
     * would have to be combined on each call).
     *
     * @param inputVector
     * @return sum-product
     */
    private double aggregate(double... inputVector) {
        double sumProduct = 0;

        if (weightVector.length != 0) {
            //ignores the last element in weightVector, the bias
            for (int i = 0; i < inputVector.length; i++) {
                sumProduct += inputVector[i] * weightVector[i];
            }
            //substract the bias
            sumProduct -= weightVector[weightVector.length - 1];
        } else {//input neuron (no weights, input is unchanged)
            sumProduct = inputVector[0];
        }
        
        return sumProduct;
    }

    //the last position in the array is used to store the bias
    private double[] weightVector;
    //store neuron output to facilitate some training algorithms
    private double output;
    private IFunction activationFunction;
}
