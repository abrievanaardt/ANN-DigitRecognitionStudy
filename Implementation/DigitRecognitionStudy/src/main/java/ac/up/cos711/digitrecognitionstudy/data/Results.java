package ac.up.cos711.digitrecognitionstudy.data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

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
            writer = new BufferedWriter(new FileWriter(experimentName + "/" + resultType + ".csv", true));
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
}
