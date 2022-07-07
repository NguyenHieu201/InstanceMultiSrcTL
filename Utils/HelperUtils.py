import torch

from Models.MultiSrcTL.Supervisor import MultiSrcTLSupervisor
from Models.InstanceMultiSrcTL.Supervisor import InstanceMultiSrcSupervisor
from Models.BaseModel.MLP import MLP

def mapping_supervisor(supervisor):
    match supervisor:
      case "InstanceMultiSrc":
        return {
          "Supervisor": InstanceMultiSrcSupervisor,
          "Config": "InstanceMultiSrc.json"
        }
