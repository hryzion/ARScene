import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

def sample_moons(batch_size, device):
    x, _ = make_moons(batch_size, noise=0.05)
    return torch.tensor(x, dtype=torch.float32, device=device)
def sample_circle(batch_size, device):
    theta = torch.rand(batch_size, device=device) * 2 * math.pi
    r = 1.0 + 0.05 * torch.randn(batch_size, device=device)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)
def sample_swiss_roll(batch_size, device):
    t = torch.rand(batch_size, device=device) * 4 * math.pi
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    data = torch.stack([x, y], dim=1)
    data = data / (4 * math.pi)  # 归一化一下
    return data

class MLPGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.net = nn.ModuleList([
            self._mlp(hidden_dim) for _ in range(4)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
    
    def forward(self,x):
        x = self.in_proj(x)
        for block in self.net:
            x = x+block(x)
        x = self.out_proj(x)
        return x

    def _mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        
        

class MiniDrift(nn.Module):
    def __init__(self, generator_config):
        super().__init__()
        
        in_dim = int(generator_config['in_dim'])
        out_dim = int(generator_config['out_dim'])
        hidden_dim = int(generator_config['hidden_dim'])

        self.generator = MLPGenerator(in_dim, out_dim, hidden_dim)

    def compute_V(self, x, y_pos, y_neg, T = 0.2):
        Npos = y_pos.size(0)

        dist_pos = torch.cdist(x,y_pos) # [Nx, Npos]
        dist_neg = torch.cdist(x,y_neg) # [Nx, Nneg]

        # print(dist_neg.shape)
        # print(dist_pos.shape)

        dist_neg += torch.eye(dist_neg.size(1), device=y_neg.device) * 1e6

        logits_pos = - dist_pos / T
        logits_neg = - dist_neg / T

        
        A = torch.cat([logits_pos, logits_neg], dim = -1) # [Nx, Npos+Nneg]
        A_row = F.softmax(A, dim = -1)
        A_col = F.softmax(A, dim = -2)
        A_soft = torch.sqrt(A_row*A_col)
        # print(A_soft)

        A_pos = A_soft[..., :Npos] # [Nx, Npos]
        # print(A_pos)
        A_neg = A_soft[..., Npos:] # [Nx, Nneg]
        # print(A_neg)

        Wpos = A_pos
        Wneg = A_neg

        Wpos*= A_neg.sum(dim=1,keepdim=True) 
        Wneg*= A_pos.sum(dim=1,keepdim=True)
        D_pos = Wpos@y_pos
        D_neg = Wneg@y_neg

        V = D_pos-D_neg
        return V
    def forward(self, y_pos):
        eps = torch.randn_like(y_pos)
        x = self.generator(eps) # [class_logit, pos, size, orientation]
        sample_q = x
        drifting_field = self.compute_V(x, y_pos, sample_q)
        x_drifted = x + drifting_field
        target = x_drifted.detach()
        loss = F.mse_loss(x, target)
        return loss


if __name__ == "__main__":
    config = {
        'in_dim' :2,
        'hidden_dim' : 1024,
        'out_dim' : 2
    }

    device = 'cuda:1' if torch.cuda.is_available() else "cpu"

    model = MiniDrift(config).to(device)
    save_dir = "./sanity_check"
    os.makedirs(save_dir, exist_ok=True)

    iters = 200000
    batch = 4096
    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for it in range(iters):
        model.train()
        y_pos = sample_swiss_roll(batch,device)
        loss = model(y_pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iter {it}, loss {loss.item():.6f}")
        if it % 5000 == 0:
            
            model.eval()
            z = torch.randn(5000, 2, device=device)
            gen = model.generator(z).cpu().detach()
            real = sample_swiss_roll(2000, device).cpu()

            plt.figure(figsize=(5,5))
            # plt.scatter(real[:,0], real[:,1], s=5, alpha=0.5, label='GT')
            plt.scatter(gen[:,0], gen[:,1], s=5, alpha=0.5, label='Generated')
            plt.legend()
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.title(f"Iter {it}")
            plt.savefig(f"{save_dir}/iter_{it}.png")
            plt.close()
        