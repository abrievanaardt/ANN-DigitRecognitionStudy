package ac.up.cos711.digitrecognitionstudy.experiment;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.util.IncorrectFileFormatException;
import ac.up.cos711.digitrecognitionstudy.data.util.StudyLogFormatter;
import ac.up.cos711.digitrecognitionstudy.data.util.TrainingTestingTuple;
import ac.up.cos711.digitrecognitionstudy.function.Identity;
import ac.up.cos711.digitrecognitionstudy.function.Sigmoid;
import ac.up.cos711.digitrecognitionstudy.function.util.NotAFunctionException;
import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.IFFNeuralNet;
import ac.up.cos711.digitrecognitionstudy.neuralnet.metric.ClassificationError;
import ac.up.cos711.digitrecognitionstudy.neuralnet.training.BackPropogation;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.FFNeuralNetBuilder;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.ThresholdOutOfBoundsException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.ZeroNeuronException;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Abrie van Aardt
 */
public class Test {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            setupLogging();
            
            Logger
                    .getLogger(Test.class.getName())
                    .log(Level.INFO, "Configuring experiment...");

            Dataset trainingSet = Dataset.fromFile("ac/up/cos711/digitrecognitionstudy/data/train");
            Dataset testingSet = Dataset.fromFile("ac/up/cos711/digitrecognitionstudy/data/t10k");
           
            IFFNeuralNet network = new FFNeuralNetBuilder()
                    .addLayer(trainingSet.getInputCount(), Identity.class)
                    .addLayer(trainingSet.getHiddenCount(), Sigmoid.class)
                    .addLayer(trainingSet.getTargetCount(), Sigmoid.class)
                    .build();

            new BackPropogation(0.06, 0.04).train(network, trainingSet);

            Logger.getLogger(Test.class.getName()).log(Level.INFO, 
                    "NN classification error is {0}%",
                    new ClassificationError(0.2).measure(network, testingSet));

        }
        catch (IOException | IncorrectFileFormatException 
                | NotAFunctionException | ZeroNeuronException 
                | UnequalInputWeightException | UnequalArgsDimensionException
                | ThresholdOutOfBoundsException ex) {
            Logger.getLogger(Test.class.getName()).log(Level.SEVERE, "", ex);
        }

    }

    private static void setupLogging() throws IOException {
        Formatter logFormatter = new StudyLogFormatter();
        Logger.getLogger(Test.class.getName()).setLevel(Level.CONFIG);
        Logger logger = Logger.getLogger("");        
        FileHandler logFileHandler = new FileHandler("study.log", true);
        FileHandler detailedLogFileHandler = new FileHandler("study.detailed.log", true);
        logFileHandler.setFormatter(logFormatter);
        detailedLogFileHandler.setFormatter(logFormatter);
        logger.addHandler(logFileHandler);
        logger.addHandler(detailedLogFileHandler);
        logger.setLevel(Level.ALL);
        logger.getHandlers()[0].setFormatter(logFormatter);
        logger.getHandlers()[0].setLevel(Level.CONFIG);//console output
        logger.getHandlers()[1].setLevel(Level.CONFIG);//normal log file
        logger.getHandlers()[2].setLevel(Level.ALL);//detailed log file
    }

}
