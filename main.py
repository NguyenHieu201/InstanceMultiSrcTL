import json
from Utils.HelperUtils import mapping_supervisor
import os

supervisor_config = mapping_supervisor("InstanceMultiSrc")
config_path = os.path.join("Config", supervisor_config["Config"])
config = json.load(open(config_path, "r"))
supervisor = supervisor_config["Supervisor"](config)
supervisor.train()