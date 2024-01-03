import torch

def get_similarity_matrix(self, a, b, eps=1e-8):
    """
    finds the cosine similarity matrix between each item of a w.r.t each item of b
    a and b are expected to be 2 dimensional (seq, hidden_dim)
    added eps for numerical stability
    source: https://stackoverflow.com/a/58144658
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt