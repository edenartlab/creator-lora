import torch

def compute_sampler_weights(labels):
    # manually cast to int for float labels for approximate weights
    # I hope it would be good enough
    label_counts = torch.bincount(torch.tensor(labels).int())
    label_weights = 1 / label_counts.float()
    weights = label_weights[labels]
    weights /= weights.sum()

    return weights.tolist()