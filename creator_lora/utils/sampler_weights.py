import torch

def compute_sampler_weights(labels):
    label_counts = torch.bincount(torch.tensor(labels))
    label_weights = 1 / label_counts.float()
    weights = label_weights[labels]
    weights /= weights.sum()

    return weights.tolist()