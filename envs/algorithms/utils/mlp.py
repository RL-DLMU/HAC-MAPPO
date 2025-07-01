import torch
import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)



        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        if torch.isnan(x).any():
            print(x)
            print('Error: input_mlp contains NaN values')
        x = self.fc1(x)
        if torch.isnan(x).any():
            print(x)
            print('Error: fc1 contains NaN values')
        for i in range(self._layer_N):
            x = self.fc2[i](x)
            if torch.isnan(x).any():
                print(x)
                print('Error: fc2 contains NaN values')
        return x
class MyLayerNorm(nn.Module):
    def __init__(self, obs_dim, eps=1e-5):
        super(MyLayerNorm, self).__init__()
        self.obs_dim = obs_dim
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False) + self.eps  # 加入 eps 防止除以零
        normalized_x = (x - mean) / torch.sqrt(variance)
        return normalized_x

class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:

            self.feature_norm = nn.LayerNorm(obs_dim)
            # self.feature_norm =MyLayerNorm(obs_dim)
        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        # torch.autograd.set_detect_anomaly(True)
        if torch.isnan(x).any():
            print(x)
            print('Error: x contains NaN values')
        if self._use_feature_normalization:
            x = self.feature_norm(x)
            if torch.isnan(x).any():
                print(x)
                print('Error: feature_norm contains NaN values')
        x = self.mlp(x)
        if torch.isnan(x).any():
            print(x)
            print('Error: mlp contains NaN values')
        return x