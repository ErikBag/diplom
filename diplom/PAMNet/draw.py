import os.path as osp
import argparse
import numpy as np
from matplotlib import pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from models_origin import PAMNet, Config
from utils import rmse, mae, spearman_corr, pearson
from datasets import TUDataset


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model, loader, device):
    model.eval()

    pred_list = []
    y_list = []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        pred_list += pred.reshape(-1).tolist()
        y_list += data.y.reshape(-1).tolist()

    pred = np.array(pred_list).reshape(-1,)
    y = np.array(y_list).reshape(-1,)
    return rmse(y, pred), mae(y, pred), spearman_corr(y, pred), pearson(y, pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=805, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='PPB-Affinity', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=2.0, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='cutoff in global layer')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint for model')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    train_dataset = TUDataset(path, name='train_val', use_node_attr=True).shuffle()
    test_dataset = TUDataset(path, name='test', use_node_attr=True)

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")

    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

    model = PAMNet(config).to(device)
    model.load_state_dict(torch.load(f'save/{args.checkpoint}'))
    test_rmse, test_mae, test_srcc, test_pcc = test(model, test_loader, device)
    model.eval()

    pred_list = []
    y_list = []

    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        pred_list += pred.reshape(-1).tolist()
        y_list += data.y.reshape(-1).tolist()

    pred = np.array(pred_list).reshape(-1,)
    y = np.array(y_list).reshape(-1,)

    mask = (y >= 0) & (y <= 20) & (pred >= 0) & (pred <= 20)
    y = y[mask]
    pred = pred[mask]

    save_dir = "./CrossValidationFigs"
    os.makedirs(save_dir,exist_ok=True)
    s = 5
    alpha = 0.5
    edgecolor = 'black'
    c = 'orange'
    linewidths = 0.5
    
    plt.figure(figsize=(4,4),dpi=300)
    plt.scatter(y.tolist(), pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
    plt.title(f'RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, SRCC: {test_srcc:.3f}, PCC: {test_pcc:.3f}', fontsize=8, fontweight='bold')
    plt.xticks(np.arange(0, 20.1, 2.5))  # от 0 до 20 с шагом 2.5
    plt.yticks(np.arange(0, 20.1, 2.5))  # аналогично для Y
    plt.axis('equal')
    plt.xlabel('Experiment')
    plt.ylabel('AI prediction')
    plt.tight_layout()
    plt.savefig( os.path.join(save_dir,'./new_split') )
    plt.show()
if __name__ == "__main__":
    main()