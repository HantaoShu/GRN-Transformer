import argparse
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--split_file", type=str, default='non')
parser.add_argument("--train_y_file", type=str, default='non')
parser.add_argument("--data_dir", type=str, default='non')
parser.add_argument("--pre_GRN_file", type=str, default='split')
parser.add_argument("--output_file", type=str, default='output.pkl')
opt = parser.parse_args()
tmp_res = []

out = pkl.load(open(opt.pre_GRN_file, 'rb'))
pred = np.array(out[0])
tmp = pd.read_csv(opt.data_dir + '/data.csv', index_col=0).index
idn_idf = {item: i for i, item in enumerate(tmp)}
a1 = pred.sum(-1, keepdims=True)
a2 = pred.sum(-2, keepdims=True)
a12 = pred.sum((-1, -2), keepdims=True)
avg = a1 * a2
avg = avg / a12
normalized = pred - avg
pred = normalized

z = np.load((f'{opt.data_dir}/train_z.npy'))
y = np.load((f'{opt.data_dir}/train_y.npy'))
trainy = np.load(f'{opt.data_dir}/{opt.train_y_file}')
train, test = pkl.load(open(f'{opt.data_dir}/{opt.split_file}', 'rb'))

train_xs, train_ys = [], []
test_xs, test_ys = [], []
for item in train:
    zitem = z[item]
    train_xs.append(pred[:, idn_idf[zitem[0]], idn_idf[zitem[1]]])
    train_ys.append(trainy[item])
for item in test:
    zitem = z[item]
    test_xs.append(pred[:, idn_idf[zitem[0]], idn_idf[zitem[1]]])
    test_ys.append(y[item])
train_xs = np.array(train_xs)
test_xs = np.array(test_xs)
f = RandomForestClassifier()
f.fit(train_xs, train_ys)
pred_out = f.predict_proba(test_xs)[:, 1]
test_auc = roc_auc_score(test_ys, pred_out)
test_auprc_ratio = average_precision_score(test_ys, pred_out) / np.mean(test_ys)
n = sum(test_ys)
test_epr = pd.DataFrame([pred_out, test_ys]).T.sort_values(0, ascending=False).iloc[:n][1].sum() / (n ** 2 / len(test_ys))
train_auc = roc_auc_score(train_ys, f.predict_proba(train_xs)[:, 1])
result = [opt.data_dir, test_auc, test_auprc_ratio, test_epr, train_auc]
pkl.dump(result, open(opt.output_file, 'wb'))
