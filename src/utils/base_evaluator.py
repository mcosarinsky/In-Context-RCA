import inspect
import src.metrics
from src.metrics import multiclass_score
import numpy as np

class BaseSegmentationEvaluator:
    """Base class for segmentation evaluation with common metric handling."""

    def __init__(self, n_classes, eval_metrics=['Dice']):
        # Check and store evaluation metrics
        eval_functions = dict(inspect.getmembers(src.metrics, inspect.isfunction))
        for metric in eval_metrics:
            if metric not in eval_functions:
                raise ValueError(f'{metric} is an invalid evaluation metric')
        
        self.eval_metrics = eval_metrics
        self.eval_functions = {metric: eval_functions[metric] for metric in eval_metrics}
        self.is_overlap = {metric: metric in src.metrics.is_overlap for metric in eval_metrics}
        self.n_classes = n_classes
        
    def evaluate(self, pred_label, true_label):
        """Evaluate the predicted labels against the ground truth."""
        scores = {}
        for eval_metric in self.eval_metrics:
            if self.n_classes == 1:
                scores[eval_metric] = self.eval_functions[eval_metric](pred_label, true_label)
            else:
                metric = self.eval_functions[eval_metric]
                scores[eval_metric] = multiclass_score(pred_label, true_label, metric, self.n_classes)
        return scores    
    
    def get_best_scores(self, candidate_scores):
        """
        Retrieve the best scores from the candidate scores based on the metric type.
        For overlap metrics, the highest score is chosen, while for distance metrics, the lowest is chosen.
        Handles multiclass cases by returning the best score for each class separately.
        """
        result_score = {}

        for metric in self.eval_metrics:
            scores = np.array(candidate_scores[metric])
            if self.is_overlap[metric]:
                # For overlap metrics, take the max score for each class
                result_score[metric] = np.max(scores, axis=0)
            else:
                # For distance metrics, take the min score for each class
                result_score[metric] = np.min(scores, axis=0)

        return result_score