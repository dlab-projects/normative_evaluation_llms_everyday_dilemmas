import numpy as np
import scipy


def cochran_armitage_trend_test(data, themes):
    # data: A 2D array where each row represents a group (based on number of themes)
    #       and columns represent [Success, Failure]
    # themes: The corresponding number of themes (or ranks for groups)
    
    # Number of groups
    k = len(themes)
    
    # Total number of observations in each group
    N = data.sum(axis=1)
    
    # Total number of successes and failures
    S = data[:, 0]  # Successes (label of interest)
    F = data[:, 1]  # Failures
    
    # Calculate the proportion of successes in each group
    p = S / N
    
    # Calculate the overall mean success probability
    p_hat = S.sum() / N.sum()
    
    # Cochran-Armitage test statistic
    numerator = np.sum((themes - np.mean(themes)) * (S - N * p_hat))
    denominator = np.sqrt(p_hat * (1 - p_hat) * np.sum((themes - np.mean(themes))**2 * N))
    
    Z = numerator / denominator
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(Z)))  # Two-sided p-value
    
    return Z, p_value