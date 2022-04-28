import argparse
import os
import pickle as pkl
import random
import time
import numpy as np
import pandas as pd
import torch
from src.model import Model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--layers", default=5, type=int, metavar="N", help="number of layers")
parser.add_argument("--embed_dim", default=64, type=int, metavar="N", help="embedding dimension")
parser.add_argument("--ffn_embed_dim", default=64, type=int, metavar="N", help="embedding dimension for FFN", )
parser.add_argument("--attention_heads", default=4, type=int, metavar="N", help="number of attention heads", )
parser.add_argument('--data_file', type=str, help='path of input scRNA-seq file.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
parser.add_argument("--save_name", type=str, default='./pretrain_output.pkl')
parser.add_argument("--dataset", type=str, default='')
parser.add_argument("--PIDC_file", type=str, default='')
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--rep", type=int, default=1)
opt = parser.parse_args()

device = torch.device('cuda')


def init_data(opt):
    data = pd.read_csv(opt.data_file, header=0, index_col=0).T
    corr = torch.FloatTensor(data.corr().fillna(0).values)
    spearmanr = torch.FloatTensor(data.corr('spearman').fillna(0).values)
    cov = torch.FloatTensor(data.cov().values)
    p = np.linalg.inv(cov + 0.001 * np.eye(len(cov)))
    p = torch.FloatTensor(p / np.max(abs(p)))
    try:
        pidc_data = pd.read_csv(opt.PIDC_file, sep='\t', header=None)
        pidc_data[0] = pidc_data[0].apply(lambda x: x.upper())
        pidc_data[1] = pidc_data[1].apply(lambda x: x.upper())
        pidc_data_max = pidc_data.dropna()[2].max()
        pidc_data = pidc_data.fillna(pidc_data_max)
        pidc_data = pidc_data.to_dict('records')
        gene_name_dict = {x.upper(): i for i, x in enumerate(data.columns)}
        pidc = np.zeros_like(corr)
        for item in pidc_data:
            pidc[gene_name_dict[item[0]], gene_name_dict[item[1]]] = item[2]
        pidc = torch.FloatTensor(pidc / np.max(pidc))
        networks = torch.stack([corr, spearmanr, cov, p, pidc], dim=-1).to(device)
    except:
        networks = torch.stack([corr, spearmanr, cov, p,corr], dim=-1).to(device) # need to delete
    data_values = data.values
    d_mask_np = (data_values != 0).astype(float)
    d_mask = torch.FloatTensor(d_mask_np)
    means = []
    stds = []
    for i in range(data_values.shape[1]):
        tmp = data_values[:, i]
        if sum(tmp != 0) == 0:
            means.append(0)
            stds.append(1)
        else:
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())

    means = np.array(means)
    stds = np.array(stds)
    stds[np.isnan(stds)] = 1
    stds[np.isinf(stds)] = 1
    means[np.isnan(stds)] = 0
    means[np.isinf(stds)] = 0
    stds[stds == 0] = 1
    data_values = (data_values - means) / (stds)
    data_values[np.isnan(data_values)] = 0
    data_values[np.isinf(data_values)] = 0
    data_values = np.maximum(data_values, -20)
    data_values = np.minimum(data_values, 20)
    data = pd.DataFrame(data_values, index=data.index, columns=data.columns)
    feat_train = torch.FloatTensor(data.values)
    return feat_train, d_mask_np, d_mask, data.columns, networks, means, stds


def train_model(opt):
    input_all, d_mask_np, d_mask, gene_name, networks, means, stds = init_data(opt)
    model = Model(opt).to(device)
    n_gene = len(gene_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.01)
    dataset = TensorDataset(input_all, d_mask, torch.LongTensor(list(range(len(input_all)))))
    if len(input_all) < opt.batchsize:
        dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=1, drop_last=False)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=1, drop_last=True)
    model.train()
    loss_save = []
    time1_start =time.time()
    for epoch in tqdm(range(opt.n_epochs)):
        loss_all = []
        model = model.cuda()
        for data, mask, idn in dataloader:
            data = torch.stack([data] * 1)
            mask = torch.stack([mask] * 1)
            optimizer.zero_grad()
            data_output = data.clone()
            data = data.to(device)
            mask_id = np.array(random.choices(range(data.shape[1] * data.shape[2]), k=data.shape[1] * data.shape[2] // 100))
            data[0, mask_id // n_gene, mask_id % n_gene] = 0
            mask_new = torch.zeros_like(mask)
            mask_new[0, mask_id // n_gene, mask_id % n_gene] = 1
            mask_new = (mask_new * mask).cuda()
            zeros = (data == 0).float()
            output = model(data, zeros, network=networks)
            mask_new = mask_new.to(device)
            loss = model.loss(output['logits'][:, :, :, 0], data_output.to(device), mask_new)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.001)
            optimizer.step()
            loss_all.append(loss)
        print(epoch, torch.stack(loss_all).mean())
        loss_save.append(torch.stack(loss_all).mean().cpu().item())
    time1_end=time.time()
    time2_start = time.time()
    model= model.cpu()
    input = input_all.clone().cpu().unsqueeze(0)
    zeros = (input == 0).float()
    networks = networks.cpu()
    with torch.no_grad():
        output = model(input, zeros, network=networks, train=False)
    time2_end = time.time()
    adj = []
    for i in range(output['row_attentions'].shape[-1]):
        adj.append(output['row_attentions'][:, :, i])
    adj = np.array(adj)
    pkl.dump([adj, loss_save], open(f'{opt.save_name}', 'wb'))
    return time1_end-time1_start,time2_end-time2_start

if __name__ == '__main__':
    opt = parser.parse_args()
    np.random.seed(opt.rep)
    torch.manual_seed(opt.rep)
    random.seed(opt.rep)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.rep)
    try:
        os.mkdir(opt.save_name)
    except:
        print('dir exist')
    if len(opt.dataset)==0:
        dataset_name = [x for x in opt.data_file.split('/') if '00' in x][0]
    else:
        dataset_name = opt.dataset

    opt.save_name = f'{opt.save_name}/normal_transformer_{dataset_name}_{opt.rep}'
    try:
        t1,t2 = train_model(opt)
        f = open(f'{opt.save_name}.time.txt','w')
        f.write(f'{t1}\n{t2}\n{float(torch.cuda.max_memory_allocated())/1024**2}')
        f.close()
    except:
        f = open(f'{opt.save_name}.time.txt','w')
        f.write(f'error')
        f.close()
