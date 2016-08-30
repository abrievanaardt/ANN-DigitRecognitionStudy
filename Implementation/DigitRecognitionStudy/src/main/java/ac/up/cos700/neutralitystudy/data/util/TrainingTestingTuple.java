package ac.up.cos700.neutralitystudy.data.util;

import ac.up.cos700.neutralitystudy.data.Dataset;

/**
 *
 * @author Abrie van Aardt
 */
public class TrainingTestingTuple {
    
    public TrainingTestingTuple(Dataset _training, Dataset _testing){
        training = _training;
        testing = _testing;
    }
    
    public Dataset training;
    public Dataset testing;
}
