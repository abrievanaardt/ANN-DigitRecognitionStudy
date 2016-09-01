package ac.up.cos711.digitrecognitionstudy.experiment;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;
import ac.up.cos711.digitrecognitionstudy.data.Results;
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
import java.io.IOException;
import java.lang.reflect.Method;
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

        try{
            setupLogging();
        } catch(IOException e){
            
        }

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "Configuring experiment...");

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "{0} simulation(s) of each technique will be run.", SIMULATIONS);

        Logger
                .getLogger(Study.class.getName())
                .log(Level.INFO, "============ Using All Inputs ============.");

        //============================ ALL ============================    
        experimentUsingAllInputs();
        //======================= Pre-processed =======================  
        experimentPrepocessed();
        //============================ SOM ============================
        experimentSOM();
    }

    private static void experimentUsingAllInputs() {
        BackPropogation backPropagation;
        try {
            for (int i = 1; i <= SIMULATIONS; i++) {

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
                        .addLayer(HIDDEN_UNITS, Sigmoid.class)
                        .addLayer(trainingset.getTargetCount(), Sigmoid.class)
                        .build();

                backPropagation = new BackPropogation(
                        ACCEPTABLE_TRAINING_ERROR,
                        LEARNING_RATE,
                        BIN_SIZE,
                        CLASSIFICATION_RIGOR,
                        MAX_EPOCH);

                backPropagation.train(network, trainingset, validationset);
                
                //consolidate results
                double trainingError = backPropagation.getTrainingError();
                double validationError = backPropagation.getValidationError();
                double generalisationError = new DefaultNetworkError().measure(network, generalisationset);
                //todo: classificationAccuracy is measured on the generalisation set, check correctness
                double classificationAccuracy = new ClassificationAccuracy(CLASSIFICATION_RIGOR).measure(network, generalisationset);
                
                //send results to disk
                Results.writeToFile("Exp_AllInputs", "E_t", trainingError);
                Results.writeToFile("Exp_AllInputs", "E_v", validationError);
                Results.writeToFile("Exp_AllInputs", "E_g", generalisationError);
                Results.writeToFile("Exp_AllInputs", "A_c", classificationAccuracy);

                Logger.getLogger(Study.class.getName()).log(Level.INFO,
                        "NN classification accuracy is {0}%", classificationAccuracy);
            }
        }
        catch (IOException | IncorrectFileFormatException | NotAFunctionException | ZeroNeuronException | UnequalInputWeightException | UnequalArgsDimensionException | ThresholdOutOfBoundsException ex) {
            Logger.getLogger(Study.class.getName()).log(Level.SEVERE, "", ex);
        }
    }
    
    private static void experimentPrepocessed(){
        
    }
    
    private static void experimentSOM(){
        
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

    private static int HIDDEN_UNITS = 300;

    private static double ACCEPTABLE_TRAINING_ERROR = 0.001;
    private static double LEARNING_RATE = 0.05;
    private static int BIN_SIZE = 10;
    private static double CLASSIFICATION_RIGOR = 0.2;
    private static int MAX_EPOCH = 20;
    private static int SIMULATIONS = 1;

}
