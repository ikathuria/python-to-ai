import pandas as pd
import statistics


def mean(x):
    """ Calculate the mean of a list of numbers.

    Args:
        x (list): list of numbers.

    Returns:
        mean of numbers.
    """
    return sum(x) / len(x)


def calc_variance(x, x_mean, use_package=True):
    """Calculate the variance of a list of numbers.

    Args:
        x (list): list of numbers.
        x_mean (float): mean of the list.
        use_package (bool): if statistics package should be used.

    Returns:
        variance, σ_x^2 = (1/(n-1)) ∑(i=1-n) (x_i-x_mean)^2
    """
    if use_package:
        return statistics.variance(x, x_mean)

    else:
        variance = 0.0
        for i in x:
            variance += (i - x_mean)**2
        return variance / (len(x) - 1)


def calc_covariance(x, x_mean, y, y_mean, use_package=True):
    """Calculate the covariance of two lists of numbers.

    Args:
        x (list): list of numbers.
        x_mean (float): mean of the list.
        y (list): list of numbers.
        y_mean (float): mean of the list.
        use_package (bool): if statistics package should be used.

    Returns:
        covariance, σ(x,y) = (1/(n-1)) ∑(i=1-n) (x_i-x_mean)(y_i-y_mean)
    """
    if use_package:
        return statistics.covariance(x, y), pd.Series(x).cov(pd.Series(y))

    else:
        covariance = 0.0
        for i, j in zip(x, y):
            covariance += (i - x_mean) * (j - y_mean)
        return covariance / (len(x) - 1)


def calc_sd(var):
    """Calculate the standard deviation for the list of numbers.

    Args:
        var (float): variance of the list of numbers.
    """
    return var**0.5


if __name__ == "__main__":
    # x = [float(f) for f in input("Enter space separated list for x: ")]
    # y = [float(f) for f in input("Enter space separated list for y: ")]
    x = [0, 1, 1, 1]
    x_mean = mean(x)

    y = [1, 1, 0, 1]
    y_mean = mean(y)

    # print(x.cov(y))

    use_package = True

    var1 = calc_variance(x, x_mean, use_package)
    var2 = calc_variance(y, y_mean, use_package)
    covar = calc_covariance(x, x_mean, y, y_mean, use_package)

    print("Mean of x: %f \nMean of y: %f" % (x_mean, y_mean))
    print("Variance of x: %f \nVariance of y: %f" % (var1, var2))
    if use_package:
        print("Covariance of `statistics`: %f \nCovariance of `pandas`: %f" % covar)
    else:
        print("Covariance:", covar)
    # print("Standard Deviation: %f" % calc_sd(var))
