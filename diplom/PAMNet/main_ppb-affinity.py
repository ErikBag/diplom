import os.path as osp
import argparse
import numpy as np
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
    if args.checkpoint != "":
        model.load_state_dict(torch.load(f'save/{args.checkpoint}'))
        train_rmse, train_mae, train_sd, train_p = test(model, train_loader, device)
        test_rmse, test_mae, test_sd, test_p = test(model, test_loader, device)
        print('Epoch: {:03d}, Train RMSE: {:.7f}, Train MAE: {:.7f}, Train SRCC: {:.7f}, Train PCC: {:.7f}\n \
            Test RMSE: {:.7f}, Test MAE: {:.7f}, Test SRCC: {:.7f}, Test P: {:.7f}'.format(0, train_rmse, train_mae, train_sd, train_p,
                                                                                    test_rmse, test_mae, test_sd, test_p))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], gamma=0.2)

    print("Start training!")
    best_test_rmse = None
    for epoch in range(args.epochs):
        model.train()

        for i, data in enumerate(tqdm(train_loader)):  
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.mse_loss(output, data.y) 
            loss.backward()
            optimizer.step()
            # if (i+1) % 8 ==0:
            #     optimizer.step()
            #     model.zero_grad()
        scheduler.step()
        
        train_rmse, train_mae, train_sd, train_p = test(model, train_loader, device)
        # print('Epoch: {:03d}, Train RMSE: {:.7f}, Train MAE: {:.7f}, Train SD: {:.7f}, Train P: {:.7f}'.format(epoch+1, 
        #                                                                             train_rmse, train_mae, train_sd, train_p))
        # test_rmse2, test_mae2, test_sd2, test_p2 = test(model, test_loader, device)
        # print('Epoch: {:03d}, Test RMSE: {:.7f}, Test MAE: {:.7f}, Test SD: {:.7f}, Test P: {:.7f}'.format(epoch+1, 
        #                                                                             test_rmse2, test_mae2, test_sd2, test_p2))
        test_rmse, test_mae, test_sd, test_p = test(model, test_loader, device)
        if best_test_rmse is None or test_rmse < best_test_rmse:
                torch.save(model.state_dict(), os.path.join('save', f"pamnet_ppb-affinity_dim12_{epoch+1}.pt"))
                best_test_rmse = test_rmse

        print('Epoch: {:03d}, Train RMSE: {:.7f}, Train MAE: {:.7f}, Train SRCC: {:.7f}, Train P: {:.7f}\n \
            Test RMSE: {:.7f}, Test MAE: {:.7f}, Test SRCC: {:.7f}, Test P: {:.7f}'.format(epoch+1, train_rmse, train_mae, train_sd, train_p,
                                                                                    test_rmse, test_mae, test_sd, test_p))

    print('Testing RMSE:', test_rmse)
    print('Testing MAE:', test_mae)
    print('Testing SD:', test_sd)
    print('Testing P:', test_p)


if __name__ == "__main__":
    main()