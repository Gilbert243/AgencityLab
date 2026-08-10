def effective_temperature(energy, dof=1):
    return energy / max(dof, 1)
