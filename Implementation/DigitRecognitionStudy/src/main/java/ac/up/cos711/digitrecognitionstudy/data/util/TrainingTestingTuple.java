package ac.up.cos711.digitrecognitionstudy.data.util;

import ac.up.cos711.digitrecognitionstudy.data.Dataset;

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
