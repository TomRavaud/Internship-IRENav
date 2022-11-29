import numpy as np


def kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha):
    """Implement an iteration of the classic Kalman filter

    Args:
        x_pred (ndarray (n, 1)): previous state prediction
        G_pred (ndarray (n, n)): previous x covariance matrix prediction
        y (ndarray (m, 1)): current measurement
        C (ndarray (m, n)): current C matrix
        Gbeta (ndarray (m, m)): current beta covariance matrix
        u (ndarray (n, 1)): current input of the system
        A (ndarray (n, n)): current A matrix
        Galpha (ndarray (n, n)): current alpha covariance matrix

    Returns:
        ndarray (n, 1), ndarray (n, n): new state prediction,
        new x covariance matrix prediction
    """
    ## Intermediate calculations ##
    # Innovation
    y_tilde = y - np.dot(C, x_pred)
    # Innovation covariance
    S = np.dot(np.dot(C, G_pred), np.transpose(C)) + Gbeta
    # Kalman gain
    K = np.dot(np.dot(G_pred, np.transpose(C)), np.linalg.inv(S))
    
    ## Correction step ##
    x_cor = x_pred + np.dot(K, y_tilde)
    G_cor = G_pred - np.dot(np.dot(K, C), G_pred)
    
    ## Prediction step ##
    x_pred = np.dot(A, x_cor) + u
    G_pred = np.dot(np.dot(A, G_cor), np.transpose(A)) + Galpha
    
    return x_pred, G_pred


def kalman_stationary(x_pred, y, C, u, A, K):
    """Implement an iteration of the classic Kalman filter
    in the stationary case

    Args:
        x_pred (ndarray (n, 1)): previous state prediction
        y (ndarray (m, 1)): current measurement
        C (ndarray (m, n)): constant C matrix
        u (ndarray (n, 1)): current input of the system
        A (ndarray (n, n)): constant A matrix
        K (ndarray (n, m)): constant pre-computed gain matrix

    Returns:
        ndarray (n, 1): new state prediction
    """
    # Correction and prediction steps are gathered in a single step
    x_pred = np.dot(A, x_pred) + np.dot(np.dot(A, K), y - np.dot(C, x_pred)) + u
    
    return x_pred
