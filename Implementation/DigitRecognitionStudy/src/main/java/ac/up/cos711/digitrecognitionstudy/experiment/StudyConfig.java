package ac.up.cos711.digitrecognitionstudy.experiment;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * todo: no exceptions if entries aren't found
 * @author Abrie van Aardt
 */
public class StudyConfig {

    public int hiddenUnits;
    public double acceptableTrainingError;
    public double learningRate;
    public int binSize;
    public double classificationRigor;
    public int maxEpoch;
    public int simulations;
    public int pixelBlockSize;

    public static StudyConfig fromFile(String expName) throws FileNotFoundException {
        StudyConfig config = new StudyConfig();
        Scanner fileScanner = new Scanner(new File("study.config"));
        fileScanner.useDelimiter("\\r|\\r\\n|\\n|\\s=\\s");

        fileScanner.findWithinHorizon(expName, 1000);
        fileScanner.next();

        while (fileScanner.hasNext()) {
            String next = fileScanner.next();
            
            if (next.length() > 0 && next.charAt(0) == '/'){
                break;
            }
            
            if (null != next) switch (next) {
                case "hiddenUnits":
                    config.hiddenUnits = fileScanner.nextInt();
                    break;
                case "acceptableTrainingError":
                    config.acceptableTrainingError = fileScanner.nextDouble();
                    break;
                case "learningRate":
                    config.learningRate = fileScanner.nextDouble();
                    break;
                case "binSize":
                    config.binSize = fileScanner.nextInt();
                    break;
                case "classificationRigor":
                    config.classificationRigor = fileScanner.nextDouble();
                    break;
                case "maxEpoch":
                    config.maxEpoch = fileScanner.nextInt();
                    break;
                case "simulations":
                    config.simulations = fileScanner.nextInt();
                    break;
                case "pixelBlockSize":
                    config.pixelBlockSize = fileScanner.nextInt();
                    break;
                default:
                    break;
            }
        }

        return config;
    }
}
