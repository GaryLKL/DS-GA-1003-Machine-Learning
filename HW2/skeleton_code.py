import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


### Assignment Owner: Tian Wang


#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test - test set, a 2D numpy array of size (num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    col_max = np.apply_along_axis(max, 0, train)
    col_min = np.apply_along_axis(min, 0, train)

    train_normalized = (train-col_min)/(col_max-col_min)
    test_normalized = (test-col_min)/(col_max-col_min)
    
    return train_normalized, test_normalized

#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    loss = 0 #Initialize the average square loss
    #TODO
    P = (np.dot(X, theta)-y)
    m = X.shape[0]

    loss = (1/m) * np.dot(P, P)
    return loss

#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    P = (np.dot(X, theta)-y)
    m = X.shape[0]

    return (2/m)*np.dot(X.T, P)


#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    e_i = np.zeros(num_features)
    for k in range(num_features):
        e_i[k] = 1
        approx_grad[k] = (compute_square_loss(X, y, theta+epsilon*e_i)-compute_square_loss(X, y, theta-epsilon*e_i))/(2*epsilon) 
        e_i[k] = 0

    return np.sqrt(sum((true_gradient-approx_grad)**2)) < tolerance

#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO
    true_gradient = gradient_func(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    e_i = np.zeros(num_features)
    for k in range(num_features):
        e_i[k] = 1
        approx_grad[k] = (objective_func(X, y, theta+epsilon*e_i)-objective_func(X, y, theta-epsilon*e_i))/(2*epsilon) 
        e_i[k] = 0

    return np.sqrt(sum((true_gradient-approx_grad)**2)) < tolerance


#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array, (num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    theta = np.zeros(num_features) #Initialize theta
    #TODO
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(1, num_step+1):
        g = compute_square_loss_gradient(X, y, theta)
        theta = theta - alpha*g

        # check
        if grad_check is True:
            assert grad_checker(X, y, theta) 

        # update
        avg_loss = compute_square_loss(X, y, theta)
        theta_hist[i] = theta
        loss_hist[i] = avg_loss

    return [theta_hist, loss_hist]

#######################################
### Backtracking line search
#Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO


#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    P = (np.dot(X, theta)-y)
    m = X.shape[0]

    return (2/m)*np.dot(X.T, P)+(2*lambda_reg*theta)

#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    #TODO
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(1, num_step+1):
        g = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha*g

        # update
        avg_loss = compute_square_loss(X, y, theta)
        theta_hist[i] = theta
        loss_hist[i] = avg_loss

    return [theta_hist, loss_hist]


#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=10**-2, num_epoch=1000):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch (num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size (num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist
    #TODO
    for i in range(num_epoch):
        shuffled_index = np.arange(X.shape[0])
        np.random.shuffle(shuffled_index)
        for step, j in enumerate(shuffled_index):
            g = compute_regularized_square_loss_gradient(X[j], y[j], theta, lambda_reg)
            theta = theta - (alpha/np.sqrt(step+1))*g

            # update
            avg_loss = compute_square_loss(X, y, theta)
            theta_hist[i][j] = theta
            loss_hist[i][j] = avg_loss

    return [theta_hist, loss_hist]

def main():
    #Loading the dataset
    print('loading the dataset')

    # shared/ridge_regression_dataset.csv
    df = pd.read_csv('ridge_regression_dataset.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    # linear question 4
    step_size = [0.5, 0.1, 0.05, 0.01]
    
    plt.figure()
    for i, s in enumerate(step_size):
        theta_list, loss_list = batch_grad_descent(X_train, y_train, alpha=s, num_step=1000, grad_check=False)
        plt.subplot(2, 2, i+1)
        plt.plot(range(0, 1001), loss_list)
        plt.xlabel("Iteration")
        plt.ylabel("Square Loss")
        plt.legend(labels=[s])
    plt.show()
    
    # ridge question 5
    
    '''
    step_size = np.logspace(-2, -5, 4, 10)
    lambda_list = np.append(np.logspace(-7, -1, 4, 10), np.logspace(0, 2, 3, 10))
    
    for s in step_size:
        for l in lambda_list:
            theta_list, loss_list = regularized_grad_descent(X_train, y_train, alpha=s, lambda_reg=l, num_step=1000)
            loss_train = loss_list[-1]
            loss_test = compute_square_loss(X_test, y_test, theta_list[-1])
            print("step size: {};".format(s), "lambda: {};".format(l),
             "training loss: {}".format(loss_train),
             "testing loss: {}".format(loss_test))

    # In the beginning, when step size is 10^-2 and lambda is between 10^-7 and 10^-3, I got much better training and testing loss
    '''
    s = 10**-2
    lambda_list = [(10**i)*j for i in np.linspace(-2, -5, 4) for j in [5, 1]]
    loss_train = []
    loss_test = []
    plt.figure(figsize=(5,5))
    for i, l in enumerate(lambda_list):
        theta_list, loss_list = regularized_grad_descent(X_train, y_train, alpha=s, lambda_reg=l, num_step=1000)
        loss_train.append(loss_list[-1])
        loss_test.append(compute_square_loss(X_test, y_test, theta_list[-1]))
    xtick = range(1, len(lambda_list)+1)
    plt.plot(xtick, loss_train)
    plt.plot(xtick, loss_test)
    plt.xticks(ticks=xtick, labels=lambda_list, rotation=90)
    plt.xlabel("Lambda")
    plt.ylabel("Square Loss")
    plt.legend(labels=["train", "test"])
    plt.show()

    # SGD question 5
    C = [0.1, 0.09, 0.08, 0.07]
    lambda_sgd = 10**-3
    plt.figure()
    for i, s in enumerate(C):
        theta_list, loss_list = stochastic_grad_descent(X, y, alpha=s, lambda_reg=lambda_sgd, num_epoch=1000)
        plt.subplot(2, 2, i+1)
        plt.plot(range(1, 1001), [epo[-1] for epo in loss_list])
        plt.xlabel("Epoch")
        plt.ylabel("Square Loss")
        plt.legend(labels=["C: "+str(s)])
    plt.show()
    
if __name__ == "__main__":
    main()
