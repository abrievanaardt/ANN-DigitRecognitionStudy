package ac.up.cos711.digitrecognitionstudy.data.util;

/**
 *
 * @author Abrie van Aardt
 */
public class IncorrectFileFormatException extends Exception {

    public IncorrectFileFormatException() {
        super();
    }
    
    public IncorrectFileFormatException(String msg){
        super(msg);
    }

    @Override
    public String getMessage() {
        return "Make sure the format of the dataset file is correct.";
    }
}
