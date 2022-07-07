import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from scipy import sparse
import osqp
import numpy as np
import tqdm
from numpy.lib.stride_tricks import sliding_window_view

# Get close data from csv file
def get_data(path):
    df = pd.read_csv(path)
    return df.close.to_numpy()


# Transform and inverse transform function
# Note: Customize to Scaler class
def transform(data, denominator):
    return data / denominator

def inverse_transform(data, denominator):
    return data * denominator


# Use for time-series data to use in regression problem
def time_series_sliding(data, sliding_window):
    y = data[sliding_window:]
    n_sample = y.shape[0]
    x = sliding_window_view(data, window_shape=sliding_window)[:n_sample]
    return x, y

# Splitting data - Only for time series
# Notice that validation data can use for grid search(etc)
# to optimize hyper parameter
def split_data(X, Y, test_ratio, val_ratio, random_state=42, shuffle=False):
    train_ratio = 1 - test_ratio - val_ratio
    # Split into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio,
                                                        random_state=random_state,
                                                        shuffle=shuffle)
    # Val split from train
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_ratio/(val_ratio+train_ratio),
                                                      random_state=random_state,
                                                      shuffle=shuffle)
    return {
        "train": [X_train, Y_train],
        "val": [X_val, Y_val],
        "test": [X_test, Y_test]
    }


def weight_data_mmd(source_data, target_data, gamma=0.01):
    n_source = source_data.shape[0]
    n_target = target_data.shape[0]

    ss_pairwise = rbf_kernel(X=source_data, gamma=gamma)
    tt_pairwise = rbf_kernel(X=target_data, gamma=gamma)
    st_pairwise = rbf_kernel(X=source_data, Y=target_data, gamma=gamma)


    # Define problem data
    P = sparse.csc_matrix(ss_pairwise / (n_source * n_source))
    q = np.sum(st_pairwise, axis=1) / (n_source * n_target) * (-1)
    A = sparse.csc_matrix(np.identity(n_source))
    l = np.zeros((n_source, ))
    u = np.ones((n_source, )) * n_source

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.0,
               verbose=False)

    # Solve problem
    res = prob.solve()
    v = res.x
    mmd = np.matmul(np.matmul(v.T, ss_pairwise), v) - 2 * np.sum(np.matmul(st_pairwise.T, v)) + np.sum(tt_pairwise)
    mmd = float(np.sqrt(mmd))
    return res.x, mmd

def weighted_loss(loss_fn):
    return lambda y, y_hat, w: (loss_fn(y, y_hat) * w).mean()

def weighted_train(net, loader, optimizer, criterion, epochs):
    for epoch in tqdm.tqdm(range(epochs)):
        for x, y, w in loader:
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, y, w)
            loss.backward()
            optimizer.step()