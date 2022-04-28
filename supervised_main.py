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
pred = np.concatenate([normalized, pred])

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
train_ys = np.array(train_ys)
train_ys = np.array(train_ys)
train_pred = []
for i in range(10):
    f = RandomForestClassifier(random_state=i)
    f.fit(train_xs, train_ys)
    pred_out = f.predict_proba(test_xs)[:, 1]
    score = roc_auc_score(test_ys, pred_out)  # ,roc_auc_score(train_ys,f.predict_proba(train_xs)[:,1])
    score2 = average_precision_score(test_ys, pred_out) / np.mean(test_ys)
    n = sum(test_ys)
    score3 = pd.DataFrame([pred_out, test_ys]).T.sample(frac=1).sort_values(0, ascending=False).iloc[:n][1].sum() / (
            n ** 2 / len(test_ys))
    score4 = roc_auc_score(train_ys, f.predict_proba(train_xs)[:, 1])
    train_pred.append(f.predict_proba(train_xs)[:, 1])

train_pred = pd.DataFrame(train_pred)

pos = np.nonzero(np.array(train_ys > 0.5))[0]
neg = np.nonzero(np.array(train_ys < 0.5))[0]
cutoff1 = train_pred[pos].mean().mean() - train_pred[pos].mean().std() * 2
cutoff2 = train_pred[neg].mean().mean() + train_pred[neg].mean().std() * 2
cutoff3 = train_pred[pos].std().mean() + train_pred[pos].std().std() * 2
cutoff4 = train_pred[neg].std().mean() + train_pred[neg].std().std() * 2
low_confidence_pos = np.nonzero(np.array(train_pred[pos].mean() < cutoff1))[0]
low_consistency_pos = np.nonzero(np.array(train_pred[pos].std() > cutoff3))[0]
del_pos = pos[list(set(low_confidence_pos) | set(low_consistency_pos))]
low_confidence_neg = np.nonzero(np.array(train_pred[neg].mean() > cutoff2))[0]
low_consistency_neg = np.nonzero(np.array(train_pred[neg].std() > cutoff4))[0]
del_neg = neg[list(set(low_confidence_neg) | set(low_consistency_neg))]
nondelete = list(set(range(len(train_ys)))-set(del_pos) - set(del_neg))
train_xs= train_xs[nondelete]
train_ys= train_ys[nondelete]
f = RandomForestClassifier(random_state=0)
f.fit(train_xs, train_ys)
pred_out = f.predict_proba(test_xs)[:, 1]
test_auc= roc_auc_score(test_ys, pred_out)  # ,roc_auc_score(train_ys,f.predict_proba(train_xs)[:,1])
test_auprc_ratio = average_precision_score(test_ys, pred_out) / np.mean(test_ys)
n = sum(test_ys)
test_epr = pd.DataFrame([pred_out, test_ys]).T.sample(frac=1).sort_values(0, ascending=False).iloc[:n][1].sum() / (
        n ** 2 / len(test_ys))
train_auc = roc_auc_score(train_ys, f.predict_proba(train_xs)[:, 1])

result = [opt.data_dir, test_auc, test_auprc_ratio, test_epr, train_auc]
pkl.dump(result, open(opt.output_file, 'wb'))
