import torch
import torch.nn as nn

class XNetLayerND(nn.Module):
    def __init__(self, in_features, num_basis):
        super(XNetLayerND, self).__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        
        self.d = nn.Parameter(torch.rand(num_basis) * 0.5 + 0.1)
        self.lambda1 = nn.Parameter(torch.randn(num_basis))
        self.lambda2 = nn.Parameter(torch.randn(num_basis))
        
        # MUDANÇA AQUI: Agora cada centro tem 'in_features' dimensões (10D)
        self.centers = nn.Parameter(torch.randn(num_basis, in_features))
        
        self.output_linear = nn.Linear(num_basis, 1)

    def forward(self, x):
        # x shape: (batch, in_features)
        # centers shape: (num_basis, in_features)
        
        # Expandimos as dimensões para cruzar todos os inputs com todos os centros
        x_exp = x.unsqueeze(1)          # Fica (batch, 1, in_features)
        c_exp = self.centers.unsqueeze(0) # Fica (1, num_basis, in_features)
        
        # MUDANÇA AQUI: Distância Euclidiana ao quadrado
        dist_sq = torch.sum((x_exp - c_exp)**2, dim=2) # Fica (batch, num_basis)
        
        # Distância radial pura (usamos 1e-8 para evitar divisão por zero no gradiente)
        dist = torch.sqrt(dist_sq + 1e-8)
        
        denom = dist_sq + self.d.unsqueeze(0)**2
        
        # Fórmula de Cauchy adaptada para o raio multidimensional
        phi = (self.lambda1.unsqueeze(0) * dist) / denom + self.lambda2.unsqueeze(0) / denom
        
        return self.output_linear(phi)