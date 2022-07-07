import os
from tkinter import N
from traceback import print_tb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from torch import optim
import torch
import matplotlib.pyplot as plt
from Models.MultiSrcTL.Dataset import get_set_and_loader
from Models.BaseModel.MLP import MLP
from Models.MultiSrcTL.Model import *
import json
from Models.MultiSrcTL.Utils import *

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# configPath = os.path.join(ROOT_DIR, "config.json")

# workspace = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, os.pardir)) 
# print(workspace)

# params = json.load(open(configPath, "r")
# print(os.path.abspath(os.path.join(ROOT_DIR, os.pardir)))
# with open("config.json", "w") as outfile:
#     json.dump({
#         "lr": 0.01,
#         "optimizer": "Adam",
#         "Source folder": "",
#         "Source domain": ["ACB, ABC"]
#     }, outfile)

class MultiSrcTLSupervisor():
    def __init__(self, config):
        pass

    def train(self):
        print(os.getcwd())
        pass

    def save(self):
        pass

    def load(self):
        pass
# denominator = 10000
# src_folder = "./Data/stock_data/data_for_teacher"
# tar_folder = "./Data/stock_data/data_for_student"

# tar_path = os.path.join(tar_folder, "GVR.csv")

# tar = get_data(tar_path)
# tar = transform(data=tar, denominator=denominator)
# tar_input, tar_output = time_series_sliding(data=tar, sliding_window=22)
# sliding_window = 22
# source_domains = []

# for file in os.listdir(src_folder):
#     src_name = file.split(sep=".")[0]
#     data_path = os.path.join(src_folder, file)
#     src = get_data(data_path)
#     src = transform(data=src, denominator=denominator)
#     src_input, src_output = time_series_sliding(data=src, sliding_window=sliding_window)
#     src_data = split_data(X=src_input, Y=src_output, val_ratio=0.2, test_ratio=0.2,
#                           shuffle=False)

#     tar_data = split_data(X=tar_input, Y=tar_output, test_ratio=0.2, val_ratio=0.2,
#                           shuffle=False)

#     res, mmd = weight_data_mmd(source_data=src_data['train'][0],
#                                target_data=tar_data['train'][0],
#                                gamma=0.01)
    
#     tmp_dataset, tmp_loader = get_set_and_loader(X=src_data['train'][0],
#                                                 Y=src_data['train'][1],
#                                                 W=res,
#                                                 batch_size=64, shuffle=False)

#     mlp = MLP(params={
#             "l": 22,
#             "p": 1
#         })

#     criterion = weighted_loss(torch.nn.MSELoss(reduction='none'))
#     optimizer = optim.Adam(mlp.parameters(), lr=0.001)
#     weighted_train(mlp, loader=tmp_loader, optimizer=optimizer,
#                    criterion=criterion, epochs=100)
#     source_domains.append({
#         "data": src_data['train'],
#         "mmd": mmd,
#         "model": mlp
#     })

#     input = torch.Tensor(tar_data['test'][0])
#     output = mlp(input) * 10000
#     ground_true = tar_data['test'][1] * 10000
#     prediction = output.detach().numpy()[:ground_true.shape[0]]
#     plt.plot(prediction, 'k', label="Prediction")
#     plt.plot(ground_true, 'r', label="Ground truth")
#     plt.savefig(f"Models/MultiSrcTL/Result/{src_name}.png")
#     plt.clf()

#     print(f"\t{mmd}")
#     print("\tRMSE: ", np.sqrt(mean_squared_error(y_pred=prediction, y_true=ground_true)))
#     print("\tMAPE: ", mean_absolute_percentage_error(y_pred=prediction, y_true=ground_true))
#     print("\tR2 score: ", r2_score(y_pred=prediction, y_true=ground_true))


# customMultiSrc = CustomMultiSrcTL(src_domain=source_domains,
#                                  BETA=0.1,
#                                  SIGMA=0.001,
#                                  p=1)
# customMultiSrc.compute_inter_src_relation_matrix()
# customMultiSrc.compute_source_target_relation()
# customMultiSrc.compute_source_weight()
# true = tar_data["test"][1] * 10000
# pred = customMultiSrc.predict(tar_data["test"][0]) * 10000
# print("finish training")
# print("\tRMSE: ", np.sqrt(mean_squared_error(y_pred=pred, y_true=true)))
# print("\tMAPE: ", mean_absolute_percentage_error(y_pred=pred, y_true=true))
# print("\tR2 score: ", r2_score(y_pred=pred, y_true=true))
# plt.plot(true, 'k', label="Ground truth")
# plt.plot(pred, 'r', label="Prediction")
# plt.legend()
# plt.show()