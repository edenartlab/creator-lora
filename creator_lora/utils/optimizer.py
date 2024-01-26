def get_lr(optimizer):
    """
    Fetches the current learning rate from a torch optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]