import torch
from Models.BaseModel.MLP import MLP

def mapping_base_model(model_config):
  model_name = model_config["model"]
  model_params = model_config["params"]
  match model_name:
    case "MLP":
      return MLP(model_params)

def mapping_optimizer(optimizer_config, model):
  optimizer_name = optimizer_config["optim"]
  optimizer_params = optimizer_config["params"]
  match optimizer_name:
    case "Adam":
      return torch.optim.Adam(model.parameters(), **optimizer_params)

def mapping_loss(loss_config):
  loss_name = loss_config["loss"]
  match loss_name:
    case "MSE":
      return torch.nn.MSELoss