import numpy as np
import torch
import skimage.transform as ski
import SimpleITK as sitk
import torch.nn.functional as F
from universeg import universeg
from sam2.build_sam import build_sam2_video_predictor

from src.utils.base_evaluator import BaseSegmentationEvaluator
from .elastix import register_label


class SAC(BaseSegmentationEvaluator):
    """Single-atlas classifier for evaluating segmentations given input image and label."""

    def __init__(self, image=None, label=None, eval_metrics=['Dice'], n_classes=None):
        super().__init__(n_classes=n_classes, eval_metrics=eval_metrics)
        self.image = image
        self.label = label

    def set_sample(self, image, label):
        """Set the reference atlas image and label."""
        self.image = image
        self.label = label

    def predict(self, atlas, elx_params):
        """Predict the accuracy of input segmentation by transferring labels."""
        candidate_scores = {metric: [] for metric in self.eval_metrics}
        result_score = {}
        
        for sample in atlas:
            img, true_label = sample['image'], sample['GT']
            
            if self.n_classes == 1:
                true_label = true_label // true_label.max()

            pred_label = register_label(img, self.image, self.label, elx_params)

            # Evaluate metrics
            scores = self.evaluate(pred_label, true_label)
            for metric in self.eval_metrics:
                candidate_scores[metric].append(scores[metric])

        # Get the best result score based on metric type (overlap/distance)
        result_score = self.get_best_scores(candidate_scores)

        return result_score
    

# (B,N,C,H,W) for bigger support dataset
class UniverSeg(BaseSegmentationEvaluator):
    """Universal segmentation class with shared evaluation logic."""

    def __init__(self, image=None, label=None, eval_metrics=['Dice'], n_classes=None):
        super().__init__(n_classes=n_classes, eval_metrics=eval_metrics)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = universeg(pretrained=True).to(self.device)
        
        if image is not None and label is not None:
            self.image = image.unsqueeze(0).to(self.device)
            self.label = label.unsqueeze(0).to(self.device)
       
    def set_sample(self, image, label):
        """Set the reference image and label."""
        self.image = image.unsqueeze(0).to(self.device)
        self.label = label.unsqueeze(0).to(self.device)
        
    @torch.no_grad()
    def predict(self, d_reference):
        """Predict the accuracy of input segmentation on reference dataset."""
        self.model.eval()
        
        candidate_scores = {metric: [] for metric in self.eval_metrics}
        result_score = {}
        
        for sample in d_reference:
            img, true_label = sample['image'].to(self.device), sample['GT'].to(self.device)

            if self.n_classes == 1:
                # Single-class prediction
                logits = self.model(img[None], self.image[None], self.label[None])[0]
                soft_pred = torch.sigmoid(logits)
                hard_pred = soft_pred.round().clip(0, 1)
            else:
                # Multi-class prediction, soft_pred is of shape (n_labels, H, W) with probability for each class
                soft_pred = self.inference_multi(img, self.image, self.label.squeeze(0)) 
                hard_pred = torch.argmax(soft_pred, dim=0) # predict most likely class for each pixel
                true_label = torch.argmax(true_label, dim=0) # convert back from one-hot to single-channel class labels
            
            # Convert predictions to numpy arrays
            pred_label_np, true_label_np = hard_pred.cpu().numpy(), true_label.cpu().numpy()

            # Evaluate metrics
            scores = self.evaluate(pred_label_np, true_label_np)
            for metric in self.eval_metrics:
                candidate_scores[metric].append(scores[metric])

        # Get the best result score based on metric type (overlap/distance)
        result_score = self.get_best_scores(candidate_scores)

        return result_score
        
    def inference_multi(self, img, supp_img, supp_label):
        soft_pred_onehot = []
        
        # Predict separately for each class including background
        for k in range(self.n_classes + 1):
            supp_label_k = supp_label[k:k+1].unsqueeze(0)  
            logits = self.model(img[None], supp_img[None], supp_label_k[None])[0][0]
            soft_pred_onehot.append(torch.sigmoid(logits))

        soft_pred_onehot = torch.stack(soft_pred_onehot)
        return F.softmax(soft_pred_onehot, dim=0)
        

class SAM2(BaseSegmentationEvaluator):
    """Universal segmentation class with shared evaluation logic."""

    def __init__(self, config, checkpoint, image=None, label=None, eval_metrics=['Dice'], n_classes=None):
        super().__init__(n_classes=n_classes, eval_metrics=eval_metrics)
        self.image = image
        self.label = label
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)

    def set_sample(self, image, label):
        """Set the reference atlas image and label."""
        self.image = image
        self.label = label
        
    def predict(self, d_reference):
        """Predict the accuracy of input segmentation on reference dataset."""
        
        candidate_scores = {metric: [] for metric in self.eval_metrics}
        result_score = {}
        
        for sample in d_reference:
            img, true_label = sample['image'], sample['GT']
            all_images = np.array([self.image, img]) # create array with [sup_img, query_img]
            inference_state = self.predictor.init_state_by_np_data(all_images)
            class_preds = []

            if self.n_classes > 1:
                for i in range(self.n_classes):
                    # Add mask for each class to first frame (sup_img)
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=i+1, mask=self.label[i+1])
                    
                out_frame_idx, out_obj_ids, out_mask_logits = next(self.predictor.propagate_in_video(inference_state, start_frame_idx=1)) # Predict masks for second frame (query_img)

                class_preds = [(out_mask_logits[i] > 0.0).squeeze().cpu().numpy() for i in range(self.n_classes)]            
                background_pred = np.logical_not(np.logical_or.reduce(class_preds)) # Compute background predictions
                class_preds.insert(0, background_pred)
                class_preds = np.stack(class_preds, axis=0)
                pred_label = np.argmax(class_preds, axis=0)
                true_label = np.argmax(true_label, axis=0)
            else:
                 _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=self.label)
                 out_frame_idx, out_obj_ids, out_mask_logits = next(self.predictor.propagate_in_video(inference_state, start_frame_idx=1))
                 pred_label = (out_mask_logits > 0.0).squeeze().cpu().numpy()

            self.predictor.reset_state(inference_state)

            # Evaluate metrics
            scores = self.evaluate(pred_label, true_label)
            for metric in self.eval_metrics:
                candidate_scores[metric].append(scores[metric])

        # Get the best result score based on metric type (overlap/distance)
        result_score = self.get_best_scores(candidate_scores)

        return result_score