import numpy as np
import torch
from sklearn.metrics import r2_score

class InstanceMultiSrc():
    def __init__(self, src_domain, BETA=0.1, SIGMA=0.001, p=1):
        self.src_domain = src_domain
        self.n_domain = len(src_domain)

        self.BETA = BETA
        self.SIGMA = SIGMA
        self.p = p

    def empirical_error(self, src_i, src_j):
        net_j = self.src_domain[src_j]['model']
        
        input_data_i = self.src_domain[src_i]["train"]["X"]

        y_pred = net_j(torch.Tensor(input_data_i)).detach().numpy()
        y_true = self.src_domain[src_i]["train"]["Y"]
        em_error = r2_score(y_true=y_true, y_pred=y_pred)
        return em_error * self.BETA

    def compute_inter_src_relation_matrix(self):
        error_matrix = np.zeros((self.n_domain, self.n_domain))
        for i in range(self.n_domain):
            for j in range(self.n_domain):
                # all empirical_error will be exp
                if i == j:
                    error_matrix[i][j] = 1 # exp(0)
                else:
                    error_matrix[i][j] = np.exp(self.empirical_error(i, j))

        relation_matrix = np.zeros((self.n_domain, self.n_domain))
        for i in range(self.n_domain):
            for j in range(self.n_domain):
                if i == j:
                    continue
                else:
                    relation_matrix[i, j] = error_matrix[i, j] / (np.sum(error_matrix[i, :]) - 1)
        self.source_matrix = relation_matrix

    def compute_source_target_relation(self):
        st_sim = np.zeros(shape=(self.n_domain, ))
        for i in range(self.n_domain):
            st_sim[i, ] = self.src_domain[i]['mmd']

        st_sim = np.exp(np.power(st_sim, self.p) * self.SIGMA * -1)
        st_sum = np.sum(st_sim)
        st_sim = st_sim / (st_sum + 1e-6)
        self.st_sim = st_sim

    def compute_source_weight(self):
        result = self.SIGMA * np.identity(self.n_domain) + (1 - self.SIGMA) * self.source_matrix
        result = np.matmul(self.st_sim.T, result)
        self.source_weight = result

    def source_domain_predict(self, data):
        predict_list = []
        for i in range(self.n_domain):
            net = self.src_domain[i]['model']
            predict_source_i = net(torch.Tensor(data)).detach().numpy()
            predict_list.append(predict_source_i)
        return predict_list

    def predict(self, test_data):
        predict_test_data = []
        n_test = test_data.shape[0]
        predict_values = self.source_domain_predict(data=test_data)
        for i in range(n_test):
            domain_learn = np.zeros((self.n_domain, ))
            predict_value = np.zeros((self.n_domain, ))
            for t in range(self.n_domain):
                predict_value[t] = predict_values[t][i]
            for k in range(self.n_domain):
                # use self.confidence interval instead
                domain_learn[k] = (np.matmul(self.source_matrix[k, :].T, predict_value)
                                      - self.source_matrix[k, k] * predict_value[k])
                domain_learn[k] = predict_value[k]
            
            result = np.sum(self.source_weight * domain_learn)
            predict_test_data.append(result)
        
        pred = np.array(predict_test_data).ravel()
        return pred