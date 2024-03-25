import torch
import torch.nn as nn

from types_ import *


class VAE(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(VAE, self).__init__()
        
        self.latent_dim = cfg['latent_dim']
        
        # Encoder
        feature_dims = [cfg['input_dim']] + cfg['hidden_dims'] + [cfg['latent_dim']*2]
        modules = []
        for i in range(len(feature_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(feature_dims[i], feature_dims[i+1]),
                    nn.ReLU()
                )
            )
        self.encoder = nn.Sequential(*modules)
        
        # Decoder
        feature_dims.reverse()
        feature_dims[0] = feature_dims[0] // 2
        modules = []
        for i in range(len(feature_dims)-2):
            modules.append(
                nn.Sequential(
                    nn.Linear(feature_dims[i], feature_dims[i+1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(feature_dims[-2], feature_dims[-1]),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)
        
        # Initailize weights
        self.initialize()
    
    def initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.encoder(x)
        mu = out[:, :self.latent_dim]
        sigma = out[:, self.latent_dim:]
        return [mu, sigma]
    
    def decode(self, z: torch.Tensor) -> List[torch.Tensor]:
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + sigma * eps
        
    def forward(self, x: torch.Tensor, loss_fn_dict: Dict[str, Any]) -> List[torch.Tensor]:
        N, C, H, W = x.size()
        
        x = x.view(N, -1)
        mu, sigma = self.encode(x)
        
        z = self.reparameterize(mu, sigma)
        
        x_hat = self.decode(z)
        x_hat = x_hat.view([N, C, H, W])
        
        return [x_hat, z]


if __name__ == '__main__':
    net = VAE(
        input_dim=28*28,
        latent_dim=2,
        hidden_dims=[512,512,32]
    )
    print(net)
    x = torch.randn([7, 1, 28, 28], dtype=torch.float32)
    x_hat, z = net(x)
    print(x_hat.size(), z.size())