package ac.up.cos700.neutralitystudy.function.util;

/**
 *
 * @author Abrie van Aardt
 */
public class NotAFunctionException extends Exception{
    @Override
    public String getMessage(){
        return "This class does not implement the IFunction interface";
    }
}
