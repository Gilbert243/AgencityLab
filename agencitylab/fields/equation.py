def field_rhs(beta, laplacian_beta, alpha=1.0, mu=1.0):
    return alpha * laplacian_beta - mu * beta
