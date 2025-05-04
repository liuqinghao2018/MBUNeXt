
from MBUNeXt.training.Trainer.nnUNetTrainer import nnUNetTrainer
from MBUNeXt.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
import torch
import torch.nn as nn
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from MBUNeXt.network.MBUNeXt.MBUNeXt import MBUNeXt


class MBUNeXtTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        network_name: str = 'MBUNeXt'#
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, network_name)

        self.initial_lr = 1e-4
        self.weight_decay = 1e-3
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 50
        self.num_val_iterations_per_epoch = 200
        self.num_epochs = 200
        self.current_epoch = 0
        self.enable_deep_supervision = False

        self.network_name_folder = network_name

    def build_network_architecture(self,
                                   plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        model = MBUNeXt(in_channel=1, out_channel=3, training=True)

        model.apply(InitWeights_He(1e-4))
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        pass