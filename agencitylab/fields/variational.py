def action(beta, grad_beta):
    return float((beta**2 + grad_beta**2).sum())
