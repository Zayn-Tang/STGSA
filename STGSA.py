import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import scipy.stats as stats


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default=r'STGSA.yaml', 
                type=str, help='the configuration to use')
args = parser.parse_args()

print(f'Starting experiment with configurations in {args.config_filename}...')
configs = yaml.load(
    open(args.config_filename), 
    Loader=yaml.FullLoader
)
args = argparse.Namespace(**configs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = np.load(r"cdata.npy")
data[np.isnan(data)] = 0
args.temporal_slot = data.shape[1]
adj = np.load("dist_mx.npy")

def get_temporal_adj(data):
    ts = data.shape[1]
    A = np.ones((ts,ts)).astype(np.bool_)
    A = 1-np.tril(A, k=-2)
    return A

def get_spearman_mat(data):
    N  = data.shape[0]
    Acor = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            # stats.spearmanr(data[i,:,0], data[j:,0])[0]
            meani = np.mean(data[i], 0)
            meanj = np.mean(data[j], 0)
            nume = np.mean((data[i] - meani)*(data[j] - meanj), 0)
            demo = np.std(data[i]-meani, 0) * np.std(data[j]-meanj, 0)
            Acor[i][j] = Acor[j][i] =  np.mean(nume/demo)
    return Acor

def heuristic_spa_adj(dist, corr, epsilon=0.6):
    N = corr.shape[0]
    Ae = np.abs(dist-corr)
    for i in range(N):
        for j in range(N):
            if Ae[i][j] > epsilon:  # 检查超参数
                dist[i][j] = 0
            else:
                corr[i][j] = 0 
    Ahs = dist + corr
    return Ahs


class STGSABlock(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.args = args
        self.dim = dim
        self.Wtemp = nn.Parameter(torch.rand((args.history,args.history)))

        self.fea_emb1 = nn.Linear(dim*2, dim*2)
        self.fea_emb2 = nn.Linear(dim*2, dim*2)


    def forward(self, X, temp):
        adapAt = temp * self.Wtemp
        spa_tem_A = torch.einsum("bmtc,tl->bmlc", torch.einsum("bntc,nm->bmtc",X, Ahs), adapAt)
        H_star = torch.concat([spa_tem_A, X],dim=-1)
        
        # GLU
        st1  = self.fea_emb1(H_star)[...,:self.dim]
        st2  = torch.sigmoid(self.fea_emb2(H_star)[...,self.dim:])
        return st1 * st2

class STGSALayer(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.args = args

        self.layers = nn.ModuleList(
            nn.ModuleList([ STGSABlock(args, dim) for i in range(args.block_per_layers) ]) for _ in range(args.layers_per_model)
        )

        self.maxpool = nn.MaxPool2d(args.pool_kernel, padding="same")

    def forward(self, X, temp):
        out = []
        for layer in self.layers:
            layerx = X
            for block in layer:
                layerx = block(layerx, temp)
            out.append(layerx)
        output, indices = torch.max(torch.stack(out, dim=-1), dim=-1)
        return output


class STGSA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Linear(args.input_dim, args.emb_dim)

        self.layers1 = STGSALayer(args, args.emb_dim)
        self.batchnorm1 = nn.BatchNorm2d(args.emb_dim)

        self.layers2 = STGSALayer(args, args.emb_dim)
        self.batchnorm2 = nn.BatchNorm2d(args.emb_dim)

        # out module
        self.act = nn.LeakyReLU(0.01)
        self.out1 = nn.Linear(args.emb_dim, 64)
        self.out2 = nn.Linear(64, args.out_dim)

    def forward(self, X, temp):
        X = self.emb(X)

        X = self.layers1(X, temp)
        X = self.batchnorm1(X.permute(0,3,1,2)).permute(0, 2, 3,1)

        X = self.layers2(X, temp)
        X = self.batchnorm2(X.permute(0,3,1,2)).permute(0, 2, 3,1)

        out = self.out2(self.act(self.out1(X)))
        return out

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        residual = torch.abs(y_pred - y_true)
        delta = self.delta
        
        loss = torch.where(residual < delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
        return loss.mean()


At = get_temporal_adj(data)
Acor = get_spearman_mat(data)
Ahs = heuristic_spa_adj(adj, Acor)
At = torch.tensor(At, dtype=torch.float32)
Ahs = torch.tensor(Ahs, dtype=torch.float32)

delta = 1.0
loss_fn = HuberLoss(delta)

model = STGSA(args)
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Parameter Nums: ", count_parameters(model))


# out = model(X, temp)
# loss = loss_fn(out, Y)






