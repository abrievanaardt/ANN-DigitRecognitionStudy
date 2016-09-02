package ac.up.cos711.digitrecognitionstudy.data.util;

/**
 *
 * @author Abrie van Aardt
 */
public class IncompatibleBlockSizeException extends Exception {
    @Override
    public String getMessage(){
        return "The data (image) width and height is not divisible by the block size specified.";
    }
}
