'''Define DeformNet
'''

import torch
from torch import nn
from networks.meta_modules import HyperNetwork
from networks.modules import SingleBVPNet

'''Adapted from the DIF-Net repository https://github.com/microsoft/DIF-Net
'''

class DeformNet(nn.Module):
    def __init__(self, latent_dim=128, model_type='sine', hyper_hidden_layers=1, hyper_hidden_features=256,
                 mlp_input_dim=3, mlp_output_dim=8, mlp_num_hidden_layers=3, mlp_hidden_features=128,
                 **kwargs):
        super().__init__()

        # latent code embedding for training subjects
        self.latent_dim = latent_dim

        # Deform-Net
        self.mlp_net = SingleBVPNet(type=model_type, mode='mlp', hidden_features=mlp_hidden_features, num_hidden_layers=mlp_num_hidden_layers, in_features=mlp_input_dim, out_features=mlp_output_dim)

        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.mlp_net)


    def forward(self, coords, embedding, **kwargs):
        hypo_params = self.hyper_net(embedding)
        output = self.mlp_net(coords, params=hypo_params)
        return output

