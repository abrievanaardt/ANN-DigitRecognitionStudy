package ac.up.cos700.neutralitystudy.experiment;

import ac.up.cos700.neutralitystudy.data.Dataset;
import ac.up.cos700.neutralitystudy.data.Pattern;
import ac.up.cos700.neutralitystudy.data.util.IncorrectFileFormatException;
import ac.up.cos700.neutralitystudy.data.util.StudyLogFormatter;
import ac.up.cos700.neutralitystudy.data.util.TrainingTestingTuple;
import ac.up.cos700.neutralitystudy.function.Identity;
import ac.up.cos700.neutralitystudy.function.Sigmoid;
import ac.up.cos700.neutralitystudy.function.util.NotAFunctionException;
import ac.up.cos700.neutralitystudy.function.util.UnequalArgsDimensionException;
import ac.up.cos700.neutralitystudy.neuralnet.IFFNeuralNet;
import ac.up.cos700.neutralitystudy.neuralnet.training.BackPropogation;
import ac.up.cos700.neutralitystudy.neuralnet.util.FFNeuralNetBuilder;
import ac.up.cos700.neutralitystudy.neuralnet.util.UnequalInputWeightException;
import ac.up.cos700.neutralitystudy.neuralnet.util.ZeroNeuronException;
import java.io.IOException;
import java.util.Iterator;
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

            Dataset dataset = Dataset.fromFile("ac/up/cos700/neutralitystudy/data/diabetes.nsds");

            TrainingTestingTuple datasets = dataset.split(0.8);

            IFFNeuralNet network = new FFNeuralNetBuilder()
                    .addLayer(dataset.getInputCount(), Identity.class)
                    .addLayer(dataset.getHiddenCount(), Sigmoid.class)
                    .addLayer(dataset.getTargetCount(), Sigmoid.class)
                    .build();

            new BackPropogation(0.2, 0.05).train(network, datasets.training);

            Iterator<Pattern> testIt = datasets.testing.iterator();
            while (testIt.hasNext()) {
                Pattern p = testIt.next();
                double[] outputs = network.classify(p.getInputs());
                System.out.println("===== Pattern ====");
                for (int i = 0; i < outputs.length; i++) {
                    System.out.print(String.format("%.3f", outputs[i]) + "\t");
                }
                System.out.println();
                for (int i = 0; i < outputs.length; i++) {
                    System.out.print(String.format("%.3f", p.getTargets()[i]) + "\t");
                }
                System.out.println();
                System.out.println();
                System.out.println();
            }

        }
        catch (IOException | IncorrectFileFormatException | NotAFunctionException | ZeroNeuronException | UnequalInputWeightException | UnequalArgsDimensionException ex) {
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
