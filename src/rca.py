import itk
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
import faiss

from src.utils.base_rca import BaseRCA
from src.models.rca_models import SAC, UniverSeg, SAM2
from src.utils.embeddings import get_embedding


class AtlasRCA(BaseRCA):
    def __init__(self, **args):
        super().__init__(**args)

        self.elx_params = itk.ParameterObject.New()

        affine_params = self.elx_params.GetDefaultParameterMap("affine", 4)
        affine_params['MaximumNumberOfIterations'] = ['500']
        self.elx_params.AddParameterMap(affine_params)

        bspline_params = self.elx_params.GetDefaultParameterMap("bspline", 1)
        bspline_params['Metric1Weight'] = ['10']
        bspline_params['MaximumNumberOfIterations'] = ['2000']
        self.elx_params.AddParameterMap(bspline_params)

        # Configure RCA classifier using SAC
        self.classifier = SAC(eval_metrics=self.eval_metrics, n_classes=self.n_classes)

    def run_evaluation(self, d_reference, d_eval):
        """
        Runs the RCA using atlas method and returns predicted scores.
        """
        results = []

        for sample in tqdm(d_eval):
            img = sample['image']
            seg = sample['seg']
            
            # Normalize when n_classes is 1
            if self.n_classes == 1:
                seg = seg // seg.max()
            
            if 'GT' in sample:
                label = sample['GT']
                if self.n_classes == 1:
                    label = label // label.max()  # Normalize label
                real_score = self.evaluate(seg, label)
            else:
                real_score = None

            # Select subset
            if self.emb_model is not None:
                selected_subset = self.select_k_closest(d_reference, img, self.k)
            else:
                selected_subset = self.select_k_random(d_reference, self.k)

            # Run reverse classifier
            self.classifier.set_sample(img, seg)
            pred_score = self.classifier.predict(selected_subset, self.elx_params)

            results.append({'Real score': real_score, 'RCA score': pred_score})

        return results
        
    
class UniverSegRCA(BaseRCA):
    def __init__(self, **args):
        super().__init__(**args)

        # Configure RCA classifier using UniverSeg
        self.classifier = UniverSeg(eval_metrics=self.eval_metrics, n_classes=self.n_classes)

    def run_evaluation(self, d_reference, d_eval):
        """
        Runs the RCA set classifier and returns predicted scores.
        """
        results = []

        for sample in tqdm(d_eval):
            img = sample['image']
            seg = sample['seg']

            if 'GT' in sample:
                label = sample['GT']
                
                # Compute real score using base class method
                if self.n_classes > 1:
                    seg_np, label_np = torch.argmax(seg, dim=0).numpy(), torch.argmax(label, dim=0).numpy()
                else:
                    seg_np, label_np = seg.numpy(), label.numpy()
                real_score = self.evaluate(seg_np, label_np)
            else:
                real_score = None

            # Select subset
            if self.emb_model is not None:
                selected_subset = self.select_k_closest(d_reference, img, self.k)
            else:
                selected_subset = self.select_k_random(d_reference, self.k)

            # Run reverse classifier
            self.classifier.set_sample(img, seg)
            pred_score = self.classifier.predict(selected_subset)

            results.append({'Real score': real_score, 'RCA score': pred_score})

        return results
        

class SAM2RCA(BaseRCA):
    def __init__(self, **args):
        super().__init__(**args)

        # Configure RCA classifier using UniverSeg
        self.classifier = SAM2(eval_metrics=self.eval_metrics, config=args['config'], checkpoint=args['checkpoint'], n_classes=self.n_classes)

    def run_evaluation(self, d_reference, d_eval):
        """
        Runs the RCA set classifier and returns predicted scores.
        """
        results = []

        for sample in tqdm(d_eval):
            img = sample['image']
            seg = sample['seg']

            if 'GT' in sample:
                label = sample['GT']

                # Compute real score using base class method
                if self.n_classes > 1:
                    seg_np, label_np = np.argmax(seg, axis=0), np.argmax(label, axis=0)
                else:
                    seg_np, label_np = seg, label
                real_score = self.evaluate(seg_np, label_np)
            else:
                real_score = None
            
            # Select subset
            if self.emb_model is not None:
                selected_subset = self.select_k_closest(d_reference, img, self.k)
            else:
                selected_subset = self.select_k_random(d_reference, self.k)

            # Run reverse classifier
            self.classifier.set_sample(img, seg)
            pred_score = self.classifier.predict(selected_subset)

            results.append({'Real score': real_score, 'RCA score': pred_score})

        return results
    
class RCA:
    def __init__(self, **args):
        # Extract the model type from args
        model_type = args.pop('model')  
        emb_model_name = args.pop('emb_model') 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if emb_model_name is not None:
            self.processor = AutoImageProcessor.from_pretrained(emb_model_name)
            self.emb_model = AutoModel.from_pretrained(emb_model_name).to(device)
        else:
            self.processor = None
            self.emb_model = None

            # Initialize the appropriate subclass and pass processor/embedding model
        if model_type == 'universeg':
            self.model = UniverSegRCA(processor=self.processor, emb_model=self.emb_model, **args)
        elif model_type == 'atlas':
            self.model = AtlasRCA(processor=self.processor, emb_model=self.emb_model, **args)
        elif model_type == 'sam2':
            self.model = SAM2RCA(processor=self.processor, emb_model=self.emb_model, **args)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self._index_initialized = False

    def initialize_index(self, d_reference):
        """
        Initialize the FAISS index using the reference dataset.
        """
        # Create FAISS index
        self.model.index = faiss.IndexFlatIP(768)
        
        print('Populating faiss index...')
        for i in tqdm(range(len(d_reference))):
            embedding = get_embedding(d_reference[i]['image'], self.emb_model, self.processor, 
                                      normalize=True, emb_method='output')
            self.model.index.add(embedding)
            
        self._index_initialized = True

    def run_evaluation(self, d_reference, d_eval):
        """
        Run the evaluation using the selected model (UniverSeg or Atlas).
        """
        # Initialize the FAISS index if not already initialized
        if not self._index_initialized and self.emb_model is not None:
            self.initialize_index(d_reference)

        print('\nRunning predictions...')
        return self.model.run_evaluation(d_reference, d_eval)