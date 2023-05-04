import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.modules import sine_init, first_layer_sine_init, last_layer_sine_init, Sine


class MLPNet(nn.Module):
    def __init__(
            self,
            init_dims,
            output_dims=1,
            dropout=None,
            dropout_prob=0.0,
            init_latent_in=(),
            init_norm_layers=(),
            weight_norm=False,
            xyz_in_all=None,
            use_tanh=False,
            latent_dropout=False,
            activation="sine",
            **kwargs
    ):
        super(MLPNet, self).__init__()

        self.xyz_dims = 3

        # Init the network structure
        dims_init = [self.xyz_dims] + init_dims + [output_dims]
        self.num_layers_init = len(dims_init)
        self.norm_layers_init = init_norm_layers
        self.latent_in_init = init_latent_in

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers_init - 1):
            in_dim = dims_init[layer]
            if layer + 1 in init_latent_in:
                out_dim = dims_init[layer + 1] - dims_init[0]
            else:
                out_dim = dims_init[layer + 1]
                if self.xyz_in_all and layer != self.num_layers_init - 2:
                    out_dim -= self.xyz_dims
            if weight_norm and layer in self.norm_layers_init:
                setattr(
                    self,
                    "lin_" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                )
            else:
                setattr(self, "lin_" + str(layer), nn.Linear(in_dim, out_dim))
            if (
                    (not weight_norm)
                    and self.norm_layers_init is not None
                    and layer in self.norm_layers_init
            ):
                setattr(self, "bn_" + str(layer), nn.LayerNorm(out_dim))

        # Activate function
        self.nl = activation
        if self.nl == "relu":
            print("ReLU")
            self.activation = nn.ReLU()
        elif self.nl == "softplus":
            print("Softplus, beta=100")
            self.activation = nn.Softplus(beta=100)
        else:
            print("Sine")
            self.activation = Sine()
            for layer in range(0, self.num_layers_init - 1):
                lin = getattr(self, "lin_" + str(layer))
                if layer == 0:
                    first_layer_sine_init(lin)
                elif layer == self.num_layers_init - 2:
                    last_layer_sine_init(lin)
                else:
                    sine_init(lin)

        # Setup tanh ouput
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        # Setup dropouts
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.dropout_prob = dropout_prob
        self.dropout = dropout


    # input: N x (L+3)
    def forward(self, input, write_debug=False):
        xyz = input[..., -3:]

        x = xyz
        # Forward the network
        for layer in range(0, self.num_layers_init - 1):
            lin = getattr(self, "lin_" + str(layer))
            if layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], -1)
            x = lin(x)
            # bn and activation
            if layer < self.num_layers_init - 2:
                if (
                        self.norm_layers_init is not None
                        and layer in self.norm_layers_init
                        and not self.weight_norm
                ):
                    print("bn")
                    bn = getattr(self, "bn_" + str(layer))
                    x = bn(x)
                x = self.activation(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        if self.use_tanh:
            print("use_tanh")
            x = self.tanh(x)
        return x

