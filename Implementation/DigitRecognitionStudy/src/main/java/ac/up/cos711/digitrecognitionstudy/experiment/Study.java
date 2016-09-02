package ac.up.cos711.digitrecognitionstudy.experiment;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.Preprocessing;
import ac.up.cos711.digitrecognitionstudy.data.Results;
import ac.up.cos711.digitrecognitionstudy.data.util.IncompatibleBlockSizeException;
import ac.up.cos711.digitrecognitionstudy.data.util.IncorrectFileFormatException;
import ac.up.cos711.digitrecognitionstudy.data.util.StudyLogFormatter;
import ac.up.cos711.digitrecognitionstudy.data.util.TrainingTestingTuple;
import ac.up.cos711.digitrecognitionstudy.function.Identity;
import ac.up.cos711.digitrecognitionstudy.function.Sigmoid;
import ac.up.cos711.digitrecognitionstudy.function.util.NotAFunctionException;
import ac.up.cos711.digitrecognitionstudy.function.util.UnequalArgsDimensionException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.IFFNeuralNet;
import ac.up.cos711.digitrecognitionstudy.neuralnet.metric.ClassificationAccuracy;
import ac.up.cos711.digitrecognitionstudy.neuralnet.metric.DefaultNetworkError;
import ac.up.cos711.digitrecognitionstudy.neuralnet.training.BackPropogation;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.FFNeuralNetBuilder;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.ThresholdOutOfBoundsException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.UnequalInputWeightException;
import ac.up.cos711.digitrecognitionstudy.neuralnet.util.ZeroNeuronException;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Scanner;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This is where my experiment is configured.
 *
 * @author Abrie van Aardt
 */
public class Study {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        try {
            setupLogging();
        }
        catch (IOException e) {

        }

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "Configuring experiment.");

        //============================ ALL ============================    
        expUsingAllInputs();
        //======================= Pre-processed =======================  
        expPrepocessed();
        //============================ SOM ============================
        expSOM();
    }

    private static void expUsingAllInputs() {

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "============ Using All Inputs ============");

        String expName = "Exp_All_Inputs";
        StudyConfig config;
        BackPropogation backPropagation;

        try {
            config = StudyConfig.fromFile(expName);

            Logger
                    .getLogger(Study.class.getName())
                    .log(Level.INFO, "Doing {0} simulation(s)", config.simulations);

            for (int i = 1; i <= config.simulations; i++) {

                Logger
                        .getLogger(Study.class.getName())
                        .log(Level.INFO, "Starting simulation {0}.", i);

                TrainingTestingTuple trainingValidationSets = Dataset
                        .fromFile("ac/up/cos711/digitrecognitionstudy/data/train")
                        .split(0.8);

                Dataset trainingset = trainingValidationSets.training;
                Dataset validationset = trainingValidationSets.testing;
                Dataset generalisationset = Dataset.fromFile("ac/up/cos711/digitrecognitionstudy/data/t10k");

                IFFNeuralNet network = new FFNeuralNetBuilder()
                        .addLayer(trainingset.getInputCount(), Identity.class)
                        .addLayer(config.hiddenUnits, Sigmoid.class)
                        .addLayer(trainingset.getTargetCount(), Sigmoid.class)
                        .build();

                backPropagation = new BackPropogation(
                        config.acceptableTrainingError,
                        config.learningRate,
                        config.binSize,
                        config.classificationRigor,
                        config.maxEpoch);

                backPropagation.train(network, trainingset, validationset);

                //consolidate results
                double trainingError = backPropagation.getTrainingError();
                double validationError = backPropagation.getValidationError();
                double generalisationError = new DefaultNetworkError().measure(network, generalisationset);
                //todo: classificationAccuracy is measured on the generalisation set, check correctness
                double classificationAccuracy = new ClassificationAccuracy(config.classificationRigor).measure(network, generalisationset);

                //send results to disk
                Results.writeToFile(expName, "E_t", trainingError);
                Results.writeToFile(expName, "E_v", validationError);
                Results.writeToFile(expName, "E_g", generalisationError);
                Results.writeToFile(expName, "A_c", classificationAccuracy);
                Results.writeToFile(expName, "Weights", network.getWeightVector());

                Logger.getLogger(Study.class.getName()).log(Level.INFO,
                        "NN classification accuracy is {0}%", classificationAccuracy);
            }
        }
        catch (IOException | IncorrectFileFormatException | NotAFunctionException | ZeroNeuronException | UnequalInputWeightException | UnequalArgsDimensionException | ThresholdOutOfBoundsException ex) {
            Logger.getLogger(Study.class.getName()).log(Level.SEVERE, "", ex);
        }
    }

    private static void expPrepocessed() {

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "============ Preprocessed ============");

        String expName = "Exp_Preprocessed";
        StudyConfig config;
        BackPropogation backPropagation;
        try {
            config = StudyConfig.fromFile(expName);

            Logger
                    .getLogger(Study.class.getName())
                    .log(Level.INFO, "Doing {0} simulation(s)", config.simulations);

            for (int i = 1; i <= config.simulations; i++) {

                Logger
                        .getLogger(Study.class.getName())
                        .log(Level.INFO, "Starting simulation {0}.", i);

                Dataset originalDataset = Dataset.fromFile("ac/up/cos711/digitrecognitionstudy/data/train");
                Dataset reducedDataset = new Preprocessing().averagePixels(originalDataset, pixelsPerDimension, config.pixelBlockSize);
                TrainingTestingTuple trainingValidationSets = reducedDataset.split(0.8);

                Dataset trainingset = trainingValidationSets.training;
                Dataset validationset = trainingValidationSets.testing;
                Dataset generalisationset = Dataset.fromFile("ac/up/cos711/digitrecognitionstudy/data/t10k");
                Dataset reducedGeneralisationset = new Preprocessing().averagePixels(generalisationset, pixelsPerDimension, config.pixelBlockSize);

                IFFNeuralNet network = new FFNeuralNetBuilder()
                        .addLayer(trainingset.getInputCount(), Identity.class)
                        .addLayer(config.hiddenUnits, Sigmoid.class)
                        .addLayer(trainingset.getTargetCount(), Sigmoid.class)
                        .build();

                backPropagation = new BackPropogation(
                        config.acceptableTrainingError,
                        config.learningRate,
                        config.binSize,
                        config.classificationRigor,
                        config.maxEpoch);

                backPropagation.train(network, trainingset, validationset);

                //consolidate results
                double trainingError = backPropagation.getTrainingError();
                double validationError = backPropagation.getValidationError();
                double generalisationError = new DefaultNetworkError().measure(network, reducedGeneralisationset);
                //todo: classificationAccuracy is measured on the generalisation set, check correctness
                double classificationAccuracy = new ClassificationAccuracy(config.classificationRigor).measure(network, reducedGeneralisationset);

                //send results to disk
                Results.writeToFile(expName, "E_t", trainingError);
                Results.writeToFile(expName, "E_v", validationError);
                Results.writeToFile(expName, "E_g", generalisationError);
                Results.writeToFile(expName, "A_c", classificationAccuracy);
                Results.writeToFile(expName, "Weights", network.getWeightVector());

                Logger.getLogger(Study.class.getName()).log(Level.INFO,
                        "NN classification accuracy is {0}%", classificationAccuracy);
            }
        }
        catch (IncompatibleBlockSizeException | IOException | IncorrectFileFormatException | NotAFunctionException | ZeroNeuronException | UnequalInputWeightException | UnequalArgsDimensionException | ThresholdOutOfBoundsException ex) {
            Logger.getLogger(Study.class.getName()).log(Level.SEVERE, "", ex);
        }
    }

    private static void expSOM() {

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "============ SOM ============");
//        String expName = "Exp_SOM";
//        StudyConfig config;
//        BackPropogation backPropagation;
//        try {
//            config = StudyConfig.fromFile(expName);

//Logger
//                    .getLogger(Study.class.getName())
//                    .log(Level.INFO, "Doing {0} simulation(s)", config.simulations);
    }

    private static void setupLogging() throws IOException {
        Formatter logFormatter = new StudyLogFormatter();
        Logger.getLogger(Study.class.getName()).setLevel(Level.CONFIG);
        Logger logger = Logger.getLogger("");
        FileHandler logFileHandler = new FileHandler("study.log", true);
        logFileHandler.setFormatter(logFormatter);
        logger.addHandler(logFileHandler);
        logger.setLevel(Level.ALL);
        logger.getHandlers()[0].setFormatter(logFormatter);
        logger.getHandlers()[0].setLevel(Level.ALL);//console output
        logger.getHandlers()[1].setLevel(Level.CONFIG);//normal log file
    }

    private static int pixelsPerDimension = 28;

}
