from Models.InstanceMultiSrcTL.Utils import get_data, preprocessing, weighted_loss, get_model_loss_optimizer, weighted_train, weight_data_mmd
from Models.InstanceMultiSrcTL.Dataset import get_set_and_loader
from Models.InstanceMultiSrcTL.Model import InstanceMultiSrc
import matplotlib.pyplot as plt

class InstanceMultiSrcSupervisor():
    def __init__(self, config):
        self.source_folder = config["source-folder"]
        self.target_folder = config["target-folder"]
        self.source_names = config["source-domain"]
        self.target_name = config["target-domain"]
        self.test_ratio = config["test-ratio"]
        self.mode = config["mode"]
        self.setting = config["setting-params"]
        self.base_model_config = config["base-model"]
        self.optimizer_config = config["optimizer"]
        self.loss_config = config["loss"]
        self.batch_size = config["batch-size"]
        self.epochs = config["epochs"]
        # Params of transfer model
        self.gamma = config["transfer-params"]["gamma"]
        self.TL_params = config["transfer-params"]["params"]
        self.source_domain = []
        
    
    def train(self):
        target_data = get_data(folder=self.target_folder, src_name=self.target_name)
        self.target_domain = preprocessing(data=target_data, name=self.target_name,
                                           test_ratio=self.test_ratio,
                                           mode=self.mode, setting=self.setting)
        for name in self.source_names:
            data = get_data(folder=self.source_folder, src_name=name)
            src_domain = preprocessing(data=data, name=name, test_ratio=self.test_ratio,
                                       mode=self.mode, setting=self.setting)
            net, optimizer, loss_fn = get_model_loss_optimizer(base_model_config=self.base_model_config,
                                                               optimizer_config=self.optimizer_config,
                                                               loss_config=self.loss_config)

            criterion = weighted_loss(loss_fn(reduction='none'))
            sample_weight, mmd = weight_data_mmd(source_data=src_domain["train"]["X"],
                                                 target_data=self.target_domain["train"]["X"],
                                                 gamma=self.gamma)

            src_dataset, src_loader = get_set_and_loader(X=src_domain["train"]["X"],
                                                         Y=src_domain["train"]["Y"],
                                                         W=sample_weight,
                                                         batch_size=self.batch_size)

            weighted_train(net=net, loader=src_loader, optimizer=optimizer,
                           criterion=criterion, epochs=self.epochs)

            src_domain["model"] = net
            src_domain["mmd"] = mmd
            self.source_domain.append(src_domain)
            
        self.model = InstanceMultiSrc(self.source_domain, **self.TL_params)
        self.model.compute_inter_src_relation_matrix()
        self.model.compute_source_target_relation()
        self.model.compute_source_weight()
        true = self.target_domain["test"]["Y"]
        pred = self.model.predict(self.target_domain["test"]["X"])
        print("finish training")
        plt.plot(true, 'k', label="Ground truth")
        plt.plot(pred, 'r', label="Prediction")
        plt.legend()
        plt.show()
            
            



    def save(self):
        pass

    def load(self):
        pass
