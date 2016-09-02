package ac.up.cos711.digitrecognitionstudy.data;

import ac.up.cos711.digitrecognitionstudy.data.util.IncompatibleBlockSizeException;
import java.util.Iterator;

/**
 *
 * @author Abrie van Aardt
 */
public class Preprocessing {

    /**
     * This is hard-coded to work with square images. Reduces every
     * blockWidthHeight x blockWidthHeight block to a single pixel that is the
     * average of the block.
     *
     * @param dataset
     * @param imageWidthHeight
     * @return the reduced dataset
     */
    public Dataset averagePixels(Dataset dataset, int imageWidthHeight, int blockWidthHeight)
            throws IncompatibleBlockSizeException {
        if (imageWidthHeight % blockWidthHeight != 0)
            throw new IncompatibleBlockSizeException();

        Dataset reducedDataset = new Dataset();
        int blocksInDimension = imageWidthHeight / blockWidthHeight;
        int numBlocks = (int) Math.pow(blocksInDimension, 2);
        reducedDataset.setInputCount(numBlocks);
        reducedDataset.setTargetCount(dataset.getTargetCount());

        double[] blocks = new double[numBlocks];
        int blockIndex = 0;
        int rowIndex = 0;
        double[] inputs;
        Iterator<Pattern> patterns = dataset.iterator();

        while (patterns.hasNext()) {
            Pattern p = patterns.next();
            inputs = p.getInputs();

            blocks[0] = inputs[0];
            blockIndex = 0;
            rowIndex = 0;

            for (int i = 1; i < inputs.length; i++) {
                if (i % blockWidthHeight == 0) {//reached end of block (horizontally)
                    if (i % imageWidthHeight == 0) {//reached end of row
                        ++rowIndex;
                        blockIndex = rowIndex / blockWidthHeight * blocksInDimension;                       
                    }
                    else
                        ++blockIndex;
                }

                blocks[blockIndex] += inputs[i];
            }

            //get average for each block
            for (int i = 0; i < blocks.length; i++) {
                blocks[i] /= Math.pow(blockWidthHeight, 2);
            }
            
            //add blocks to the reduced dataset
            Pattern reducedPattern = new Pattern();
            reducedPattern.setInputs(blocks);
            reducedPattern.setTargets(p.getTargets());
            reducedDataset.addPattern(reducedPattern);
            blocks = new double[numBlocks];

        }
        return reducedDataset;
    }
}
