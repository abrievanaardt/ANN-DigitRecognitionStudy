package ac.up.cos711.digitrecognitionstudy.data;

import ac.up.cos711.digitrecognitionstudy.data.util.IncorrectFileFormatException;
import ac.up.cos711.digitrecognitionstudy.data.util.TrainingTestingTuple;
import java.io.DataInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class representing a dataset for neural network training. Target values 
 * are scaled to 0.1 and 0.9, instead of 0 and 1. Inputs are scaled to [-1, 1]
 *
 * @author Abrie van Aardt
 */
public class Dataset implements Iterable {

    public Dataset() {
    }

    /**
     *
     *
     * @param resourceName the prefix of the dataset name
     * @return The in-memory Dataset object
     * @throws FileNotFoundException
     * @throws IncorrectFileFormatException
     */
    public static Dataset fromFile(String resourceName)
            throws FileNotFoundException, IncorrectFileFormatException, IOException {

        Dataset dataset = new Dataset();
        ClassLoader classLoader = Dataset.class.getClassLoader();
        DataInputStream pixelStream = new DataInputStream(classLoader.getResourceAsStream(resourceName + "-images.idx3-ubyte"));
        DataInputStream labelStream = new DataInputStream(classLoader.getResourceAsStream(resourceName + "-labels.idx1-ubyte"));

        if (pixelStream == null || labelStream == null)
            throw new FileNotFoundException();
        
        //verify file format
        if (pixelStream.readInt() != 2051
                || labelStream.readInt() != 2049)
            throw new IncorrectFileFormatException();

        int numberOfPatterns = pixelStream.readInt();
        if (numberOfPatterns != labelStream.readInt())
            throw new IncorrectFileFormatException("Number of items in idx1 and idx3 fiels do not correspond.");

        //demarshal the data patterns
        int numberOfRows = pixelStream.readInt();
        int numberOfColumns = pixelStream.readInt();
        dataset.inputCount = numberOfRows * numberOfColumns;
        //targetCount already set using TARGET_COUNT

        double[] inputs = new double[dataset.inputCount];
        double[] targets = new double[dataset.targetCount];

        for (int i = 0; i < numberOfPatterns; i++) {
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = scale(pixelStream.readUnsignedByte());
            }

            digitToTargets(labelStream.readUnsignedByte(),targets);
            
            Pattern p = new Pattern();
            p.setInputs(inputs);
            p.setTargets(targets);
            dataset.data.add(p);
        }

        Logger logger = Logger.getLogger(Dataset.class.getName());
        logger.log(Level.INFO, "Loaded {3} pattern(s) with {1} input(s) "
                + "and {2} class(es) from dataset: {0}.", new Object[]{
                    resourceName.substring(resourceName.lastIndexOf('/') + 1),
                    dataset.inputCount,
                    dataset.targetCount,
                    dataset.size()
                });

        dataset.shuffle();

        return dataset;
    }

    @Override
    public Iterator<Pattern> iterator() {
        return data.iterator();
    }

    /**
     * Splits the dataset into a training and testing set. This method provides
     * respective views for the training and testing dataset. Both views are
     * backed by the same underlying dataset. trainingRatio is the proportion of
     * the dataset that will be dedicated to training patterns.
     *
     * @param trainingRatio
     * @return TrainingTestingTuple
     */
    public TrainingTestingTuple split(double trainingRatio) {
        Dataset training = new Dataset();
        Dataset testing = new Dataset();

        training.inputCount = inputCount;
        training.targetCount = targetCount;
        testing.inputCount = inputCount;
        testing.targetCount = targetCount;

        int trainingUpperIndex = (int) (trainingRatio * data.size());

        training.data = data.subList(0, trainingUpperIndex);
        testing.data = data.subList(trainingUpperIndex, data.size());

        logger.log(Level.INFO, "Using {0}"
                + "% of the patterns for training and the remainder for "
                + "validation.", String.format("%.2f", trainingRatio * 100));

        return new TrainingTestingTuple(training, testing);
    }

    public Dataset shuffle() {
        Collections.shuffle(data, random);
        return this;
    }

    /**
     * 
     * @return the number of patterns occurring in this dataset.
     */
    public int size() {
        return data.size();
    }

    public int getInputCount() {
        return inputCount;
    }

    public int getTargetCount() {
        return targetCount;
    }
    
    public void setInputCount(int count){
        inputCount = count;
    }
    
    public void setTargetCount(int count){
        targetCount = count;
    }
    
    public Pattern getPatternAt(int index){
        Pattern p = data.get(index);
        
        Pattern copy = new Pattern();
        copy.setInputs(p.getInputs());
        copy.setTargets(p.getTargets());
        
        return copy;
    }
    
    public void setPattern(int index, Pattern p){
        Pattern copy = new Pattern();
        copy.setInputs(p.getInputs());
        copy.setTargets(p.getTargets());
        
        data.set(index, copy);
    }
    
    public void addPattern(Pattern p){
        Pattern copy = new Pattern();
        copy.setInputs(p.getInputs());
        copy.setTargets(p.getTargets());
        
        data.add(copy);
    }

    private static double scale(int input) {        
        return (input / 255.0) * 2.0 - 1.0;
    }

    private static double[] digitToTargets(int digit, double[] targets) {
        for (int i = 0; i < targets.length; i++) {
            if (digit == i)
                targets[i] = 0.9;
            else
                targets[i] = 0.1;
        }
        return targets;
    }

    private List<Pattern> data = new ArrayList<>();
    private int inputCount;
    private int targetCount = 10;//10 digits to classify
    private Random random = new Random(System.nanoTime());
    private Logger logger = Logger.getLogger(getClass().getName());
}
