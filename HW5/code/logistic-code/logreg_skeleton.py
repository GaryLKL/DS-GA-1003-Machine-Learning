import numpy as np
import scipy

def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''

    n = len(y)
    exp1 = 1
    exp2 = [np.exp(y[i] * v) for i, v in enumerate(np.dot(x, theta))]
    return np.sum(np.logaddexp(exp1, exp2))/n + l2_param*np.sum(np.square(theta))
    
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
        