import seaborn as sns
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from .data_transforms import process_img

def select_k_random(dir, k=5):
    data_dir = dir / 'segs'
    labels_dir = dir / 'masks'
    seg_files = os.listdir(data_dir)
    unique_names = list(set([filename.split('_epoch')[0] for filename in seg_files]))
    n = len(unique_names)

    indices = np.random.permutation(n)[:k]

    to_tensor = transforms.ToTensor()
    selected_labels = []
    selected_segs = defaultdict(list)

    for idx in indices:
        img_name = unique_names[idx]
        # Handle the case where img_name contains underscores or not
        if 'irca' in img_name:
            label_path = os.path.join(labels_dir, img_name.replace('_', '/', 1) + '.png')

        elif '_' in img_name and 'ISIC' not in img_name:
            label_path = os.path.join(labels_dir, img_name.replace('_', '/').split('.')[0] + '.png')
        else:
            label_path = os.path.join(labels_dir, img_name + '.png')

        # Check if the label_path exists before processing
        if os.path.exists(label_path):
            label_tensor = to_tensor(process_img(label_path, 256, is_seg=True)) * 255
            selected_labels.append(label_tensor)
        else:
            print(f"Warning: File {label_path} not found.")
            selected_labels.append(None)

        for seg_path in seg_files:
            if img_name in seg_path:
                seg = Image.open(os.path.join(data_dir, seg_path)).convert('L')
                seg_tensor = to_tensor(seg) * 255

                epoch = seg_path.split('epoch')[1].split('.')[0]
                selected_segs[f'epoch_{epoch}'].append(seg_tensor)

    return selected_labels, selected_segs


def visualize_tensors(tensors, k, title=None):
    M = len(tensors)  # Number of keys (rows)
    cols = k  # Number of columns is the number of samples per key
    rows = M  # Each key corresponds to a row

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d * cols, d * rows))

    if rows == 1:
        axes = axes.reshape(1, cols)
    if cols == 1:
        axes = axes.reshape(rows, 1)

    for row, (grp, tensor_list) in enumerate(tensors.items()):
        for col, tensor in enumerate(tensor_list[:k]):
            ax = axes[row, col]
            x = tensor.detach().cpu().numpy().squeeze()
            if len(x.shape) == 2:
                ax.imshow(x, vmin=x.min(), vmax=x.max())
            else:
                ax.imshow(np.transpose(x, (1, 2, 0)))

            if col == 0:
                ax.set_ylabel(grp, fontsize=16)

    # Hide any empty subplots
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()
    
    
def dice_histogram(scores, n_bins, title=None):
    weights = np.ones_like(scores) / len(scores)
    fig, ax = plt.subplots()
    ax.hist(scores, bins=n_bins, edgecolor='black', weights=weights)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(title)
    ax.set_xlabel('Dice Score')
    ax.set_ylabel('Probability Density')
    
    return fig, ax

def plot_score(ax, real_scores, pred_scores, unit_scale=True):
    # Convert to numpy array
    real_scores = np.array(real_scores)
    pred_scores = np.array(pred_scores)

    # Calculate correlation and MAE
    corr = np.corrcoef(real_scores, pred_scores)[0, 1]
    mae = np.mean(np.abs(real_scores - pred_scores))
    
    sns.scatterplot(x=real_scores, y=pred_scores, ax=ax)
    ax.annotate(
        f'Corr: {corr:.2f}\nMAE: {mae:.2f}',
        xy=(0.05, 0.9),  # Adjusted y-coordinate for annotation box
        xycoords='axes fraction',
        fontsize=11,  # Increased fontsize for better readability
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
    )

    if unit_scale:
        ticks = np.arange(0.0, 1.1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    else:
        max_limit = max(real_scores.max(), pred_scores.max()) + 5
        ax.set_xlim([0, max_limit])
        ax.set_ylim([0, max_limit])
        
    ax.set_aspect('equal') 
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicted')


def plot_scores_multi(ax, real_scores, pred_scores, class_names=None, unit_scale=True):
    """
    Plot multi-class scores on the same axis, reporting MAE and correlation for each class.
    real_scores and pred_scores are expected to be 2D arrays with shape (n_samples, n_classes).
    """
    n_classes = real_scores.shape[1]  # Number of classes
    colors = sns.color_palette("colorblind", n_classes)  # Generate a color for each class

    # Default class names if not provided
    if class_names is None:
        class_names = [f'Class {i + 1}' for i in range(n_classes)]

    for i in range(n_classes):
        real_class_scores = real_scores[:, i]
        pred_class_scores = pred_scores[:, i]

        # Calculate correlation and MAE
        corr = np.corrcoef(real_class_scores, pred_class_scores)[0, 1]
        mae = np.mean(np.abs(real_class_scores - pred_class_scores))

        # Plot the class scores with different colors
        sns.scatterplot(x=real_class_scores, y=pred_class_scores, ax=ax, color=colors[i],
                        label=f'{class_names[i]} (Corr: {corr:.2f}, MAE: {mae:.2f})')

    # Set axis limits
    if unit_scale:
        ticks = np.arange(0.0, 1.1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    else:
        max_limit = max(real_scores.max(), pred_scores.max()) + 5
        ax.set_xlim([0, max_limit])
        ax.set_ylim([0, max_limit])

    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicted')
    ax.legend(loc='upper left', fancybox=True)


def plot_results(eval_results, **kwargs):
    sns.set_theme(style="whitegrid")

    num_plots = len(eval_results)
    n_cols = kwargs.get('n_cols', 3)
    n_rows = math.ceil(num_plots / n_cols)
    figsize = kwargs.get('figsize', 5)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize * n_cols, figsize * n_rows))
    axes = axes.flatten()
    unit_scale = kwargs.get('unit_scale', True)
    multiclass = kwargs.get('is_multiclass', False)

    # Get class names from kwargs or set default
    class_names = kwargs.get('class_names', None)

    # Plot each result
    titles = kwargs.get('titles', [None] * num_plots)
    for ax, (res, title) in zip(axes, zip(eval_results, titles)):
        real_scores = np.array(res['Real'])
        pred_scores = np.array(res['Predicted'])

        if multiclass:
            plot_scores_multi(ax, real_scores, pred_scores, class_names=class_names, unit_scale=unit_scale)
        else:
            plot_score(ax, real_scores, pred_scores, unit_scale=unit_scale)
        
        ax.set_title(title)

    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    suptitle = kwargs.get('suptitle', 'Predicciones multiclase variando N_test')
    plt.suptitle(suptitle, fontsize=20)

    # Adjust layout to make room for the supertitle and increase space between rows
    plt.tight_layout(pad=2.0, h_pad=2.0, rect=[0, 0, 1, 0.99])

    return fig


def plot_residuals(real, predicted):
    real = np.array(real)
    predicted = np.array(predicted)

    # Calculate residuals
    residuals = real - predicted

    # Plot residuals
    fig, ax = plt.subplots()
    ax.scatter(predicted, residuals)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    plt.tight_layout()

    return fig, ax

def print_statistics(real, predicted):
    real = np.array(real)
    predicted = np.array(predicted)

    # Calculate residuals
    residuals = real - predicted

    # Compute statistics
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    std_residual = np.std(residuals)
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)

    # Print statistics
    print(f'Mean Residual: {mean_residual:.4f}')
    print(f'Median Residual: {median_residual:.4f}')
    print(f'Standard Deviation of Residuals: {std_residual:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')