from torch.utils.data import Subset
import numpy as np
from src.utils.embeddings import get_embedding
from src.utils.base_evaluator import BaseSegmentationEvaluator

class BaseRCA(BaseSegmentationEvaluator):
    def __init__(self, **args):
        # Initialize the base class to handle eval metrics
        super().__init__(args['n_classes'], args.get('eval_metrics', ['Dice']))

        # Extract and set common parameters
        self.k = args.get('n_test', 24)
        self.processor = args['processor']  
        self.emb_model = args['emb_model']

    def select_k_closest(self, dataset, image, k):
        """
        Select the k closest images in the dataset to the input image in an embedding space.
        """
        img_embedding = get_embedding(image, self.emb_model, self.processor)
        _, indices = self.index.search(img_embedding, k) 
        subset = Subset(dataset, indices.reshape(-1))
        
        return subset
    
    def select_k_random(self, dataset, k):
        idxs = np.random.permutation(len(dataset))[:k]
        subset = Subset(dataset, idxs)
        return subset

    def run_evaluation(self, d_reference, d_eval):
        """
        Runs the RCA set classifier and returns predicted scores.
        This method is generalized and will be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")