import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import sparse
import osqp
import numpy as np
import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import torch
from Utils.Mapping import mapping_base_model, mapping_optimizer, mapping_loss

# Get close data from csv file
def get_data(folder, src_name):
    path = os.path.join(folder, f"{src_name}.csv")
    df = pd.read_csv(path)
    data = df.close.to_numpy()
    return data


def time_series_processing(data, mode, setting):
    match mode:
        case "one-day":
            sliding_window = setting["seq_len"]
            y = data[sliding_window:]
            n_sample = y.shape[0]
            x = sliding_window_view(data, window_shape=sliding_window)[:n_sample]
            return {
                "X": x.astype(np.float32),
                "Y": y.reshape(-1, 1).astype(np.float32)
            }


def preprocessing(data, name, test_ratio, mode, setting):
    train, test = train_test_split(data, test_size=test_ratio, shuffle=False)

    train = time_series_processing(data=train, mode=mode, setting=setting)
    test = time_series_processing(data=test, mode=mode, setting=setting)
    # Scale data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(train["X"])
    y_scaler.fit(train["Y"])
    train["X"] = x_scaler.transform(train["X"])
    train["Y"] = y_scaler.transform(train["Y"])
    test["X"] = x_scaler.transform(test["X"])
    test["Y"] = y_scaler.transform(test["Y"])

    return {
        "name": name,
        "train": train,
        "test": test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler
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


def get_model_loss_optimizer(base_model_config, optimizer_config, loss_config):
    net = mapping_base_model(base_model_config)
    optimizer = mapping_optimizer(optimizer_config, net)
    loss_fn = mapping_loss(loss_config)
    return net, optimizer, loss_fn

def weighted_loss(loss_fn):
    return lambda y, y_hat, w: (loss_fn(y, y_hat) * w).mean()

def weighted_train(net, loader, optimizer, criterion, epochs):
    for epoch in tqdm.tqdm(range(epochs)):
        for x, y, w in loader:
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(torch.reshape(output, y.shape), y, w)
            loss.backward()
            optimizer.step()