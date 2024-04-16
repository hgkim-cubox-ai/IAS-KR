import torch
import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(FCModel, self).__init__()
        """
        cfg: Dict[str, Any]
        """
        feature_dim = cfg['Data']['patch_size']**2 * 3
        self.patch_layer = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, 1024, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 256, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(cfg['Data']['n_patches'])
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        N, P = x.shape[:2]      # (N, P, C, H, W)
        x = x.view(N, P, -1)    # (N, P, CxHxW)
        x = x.permute(0, 2, 1)  # (N, CxHxW, P)
        x = self.patch_layer(x)
        x = torch.squeeze(x, -1)
        x = self.fc(x)
                
        return x


if __name__ == '__main__':
    net = FCModel({'Data':{'patch_size': 64,
                           'n_patches': 8}})
    input = torch.randn([7, 8, 3, 64, 64])
    output = net(input)
    print(output.size())