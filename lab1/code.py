import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest

def do_stuff():

    # Load data
    data = pd.read_csv('../data/BostonHousing.csv')
    target = 'medv'
    data = pd.read_csv('../data/Admission.csv')
    target = 'Chance of Admit'
    data = pd.read_csv('../data/world-happiness-report-2021.csv', usecols =[2,6,7,8,9,10,11])
    target = 'Ladder score'

    # Separate descriptive variable and target variable 
    X = data.drop(target, axis=1)
    y = data[target]

    # Add constant to descriptors, to act as intercept (value when all descriptors are zero)
    X = sm.add_constant(X)

    # Create linear regression model using GLM function
    model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()

    # Print summary of model
    print(model.summary())

    ## Plot residuals
    #Should be normal along y, centered on y=0
    resid = model.resid_response
    plt.scatter(model.fittedvalues, resid)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    # Perform normality test on residuals

    # SW Null Hyp: Guassian origin. p : probability of finding this data (0.01->1%).
    stat, p = shapiro(resid)
    print('Shapiro-Wilk Test for Normality of Residuals')
    print('Statistic: %.3f, p-value: %.3f' % (stat, p))

    # KS statistic: Max. distance between cumulative distrib.
    stat, p = kstest(resid, 'norm')
    print('Kolmogorov-Smirnov Test for Normality of Residuals')
    print('Statistic: %.3f, p-value: %.3f' % (stat, p))

    #See by yourself. Is this normal?

    plt.hist(resid, density=True)
    plt.title('Residual Distribution')
    plt.show()


if __name__ == "__main__":
    print(sys.argv[1:])
    do_stuff()
