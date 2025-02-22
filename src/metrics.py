import medpy.metric.binary as metrics
import numpy as np

is_overlap = {
    'Dice': True
}

def multiclass_score(result, reference, metric, num_classes):
    scores = []
    
    for i in range(1, num_classes+1): 
        result_i, reference_i = (result == i).astype(int), (reference==i).astype(int)
        scores.append(metric(result_i, reference_i))
    
    return scores

def Hausdorff(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.hd(result, reference)

def ASSD(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.assd(result, reference)

def Dice(result, reference):
    return metrics.dc(result, reference)

def compute_scores(data, num_classes, metric=Dice):
    scores = []
    
    for sample in data:
        result, reference = sample['seg'], sample['GT']
        score = multiclass_score(result, reference, metric, num_classes)
        scores.append(score)
    
    return np.array(scores)


def sample_N(scores, N, n_buckets=10, samples_per_bucket=None):
    """
    Sample a specific number of items from each bucket based on `N` and `n_buckets`.
    
    Args:
        scores: Array of scores.
        N: Total number of samples to take.
        n_buckets: Number of buckets to create between 0 and 1 (default is 10).
        samples_per_bucket: List of integers specifying how many samples to draw from each bucket.
                            If None, it will evenly distribute `N` samples across the buckets.
                            The length of the list will determine the number of buckets.
    Returns:
        np.array: Indices of the sampled items.
    """
    if samples_per_bucket is not None:
        n_buckets = len(samples_per_bucket)

    # Create `n_buckets` equally spaced bins between 0 and 1
    bins = np.linspace(0, 1, n_buckets + 1) 
    bucket_indices = np.digitize(scores, bins, right=False) - 1  # Divide scores into bins
    buckets = [np.where(bucket_indices == i)[0] for i in range(n_buckets)]  # Indices for each bucket

    # If `samples_per_bucket` is not provided, distribute `N` samples evenly across buckets
    if samples_per_bucket is None:
        samples_per_bucket = [N // n_buckets] * n_buckets
        remainder = N % n_buckets
        # Distribute the remainder samples across the last few buckets
        for i in range(n_buckets - remainder, n_buckets):
            samples_per_bucket[i] += 1

    sampled_indices = []

    for i in range(n_buckets):
        num_samples = samples_per_bucket[i]
        
        if len(buckets[i]) >= num_samples:  # If there are enough elements in the bucket
            sampled_indices.extend(np.random.choice(buckets[i], size=num_samples, replace=False))
        else:
            sampled_indices.extend(buckets[i]) # If not enough elements, sample all available 

    return np.array(sampled_indices)