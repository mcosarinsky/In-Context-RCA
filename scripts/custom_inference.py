import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_dir = os.path.join(parent_dir, 'sam2')
sys.path.extend([parent_dir, sam2_dir])

import argparse
import torchvision
from src.utils.io import to_json
from src.rca import RCA
from src.datasets.segmentation_2d import CustomImageDataset
from src.utils.data_transforms import ToTensor, OneHot

def main():
    parser = argparse.ArgumentParser(description="Run custom RCA inference on user-supplied data")
    parser.add_argument('--ref_dataset', type=str, required=True, help='Path to reference dataset folder')
    parser.add_argument('--eval_dataset', type=str, required=True, help='Path to the segmentations to evaluate and images they originate from')
    parser.add_argument('--classifier', type=str, required=True, help='Classifier used to run inference')
    parser.add_argument('--output_file', type=str, required=True, help='Output file to store results')
    parser.add_argument('--n_test', type=int, default=16, help='Number of reference images to use')
    parser.add_argument('--n_classes', type=int, required=True, help='Number of segmentation classes')
    parser.add_argument('--emb_model', type=str, default='facebook/dinov2-base', help='Embedding model')
    parser.add_argument('--config', type=str, default='sam2_hiera_t', help='SAM2 model config (if applicable)')
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['Dice', 'Hausdorff', 'ASSD'],
                        help='List of evaluation metrics to compute (optional)')
    args = parser.parse_args()
    
    # If no embedding model is provided, the n_test reference images are selected at random
    if args.emb_model == 'None':
        args.emb_model = None

    transforms_list = []

    # Apply transforms based on the classifier
    if args.classifier == 'universeg':
        transforms_list.extend([ToTensor(), OneHot(n_classes=args.n_classes)])
    elif args.classifier == 'sam2':
        transforms_list.append(OneHot(n_classes=args.n_classes))
        
    transforms = torchvision.transforms.Compose(transforms_list) if transforms_list else None

    ref_dataset = CustomImageDataset(args.ref_dataset, type='reference', target_size=256, transform=transforms)
    eval_dataset = CustomImageDataset(args.eval_dataset, type='eval', target_size=256, transform=transforms)

    # Set up RCA parameters.
    rca_args = {
        'eval_metrics': args.eval_metrics,
        'n_test': args.n_test,
        'n_classes': args.n_classes,
        'emb_model': args.emb_model
    }

    if args.classifier == 'sam2':
        sam2_checkpoints = 'sam2/checkpoints/'
        model_type_dict = {'sam2_hiera_t':'sam2_hiera_tiny',
                           'sam2_hiera_s':'sam2_hiera_small',
                           'sam2_hiera_b+':'sam2_hiera_base_plus',
                           'sam2_hiera_l':'sam2_hiera_large'}
        for k, v in model_type_dict.items():
            model_type_dict[k] = os.path.join(sam2_checkpoints, f'{v}.pt')
        rca_args['config'] = f"{args.config}.yaml"
        rca_args['checkpoint'] = model_type_dict[args.config]

    # Initialize RCA classifier.
    rca_clf = RCA(model=args.classifier, **rca_args)
    preds = rca_clf.run_evaluation(ref_dataset, eval_dataset)  
    
    # Save predictions.
    to_json(preds, args.output_file)

if __name__ == "__main__":
    main()