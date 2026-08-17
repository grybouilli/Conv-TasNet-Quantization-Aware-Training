import numpy as np

def mean_and_std(results):
    """
    Compute mean and standard deviation (sample SD) of experimental runs.

    Parameters
    ----------
    results : list or array-like
        Results from multiple runs (e.g., accuracies).

    Returns
    -------
    mean : float
        Mean of the results.
    std : float
        Sample standard deviation.
    """
    results = np.asarray(results, dtype=float)
    mean = results.mean()
    std = results.std(ddof=1)  # ddof=1 -> sample standard deviation
    return mean, std


# Example usage
runs = [73.94, 74.03, 74.18, 73.93, 74.14]
mean, std = mean_and_std(runs)

print(f"Result: {mean:.2f} ± {std:.2f}")
