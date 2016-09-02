package ac.up.cos711.digitrecognitionstudy.data;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

/**
 *
 * @author Abrie van Aardt
 */
public class Results {

    /**
     * Appends the value to a file with the relative path:
     * experimentName/resultType
     *
     * @param experimentName
     * @param resultType
     * @param value
     * @throws IOException
     */
    public static void writeToFile(String experimentName, String resultType, double value) throws IOException {
        BufferedWriter writer = null;
        try {
            File directory = new File(experimentName + "/" + resultType + ".csv");
            directory.getParentFile().mkdirs();
            writer = new BufferedWriter(new FileWriter(directory, true));
            writer.write(Double.toString(value));
            writer.newLine();
            writer.flush();
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                }
                catch (IOException e) {
                }
            }
        }
    }
    
    public static void writeToFile(String experimentName, String resultType, double[] values) throws IOException {
        BufferedWriter writer = null;
        try {
            File directory = new File(experimentName + "/" + resultType + ".dat");
            directory.getParentFile().mkdirs();
            writer = new BufferedWriter(new FileWriter(directory, true));
            writer.write(Arrays.toString(values));
            writer.newLine();
            writer.flush();
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                }
                catch (IOException e) {
                }
            }
        }
    }
}
