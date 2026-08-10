def second_law_check(entropy_series):
    return all(x2 >= x1 for x1, x2 in zip(entropy_series, entropy_series[1:]))
