import numpy as np
import sys
sys.path.append('/Users/diarshakimov/Downloads/CMPT-310/q3')
import q3 as q3

all_data = q3.load_auto_data('/Users/diarshakimov/Downloads/CMPT-310/q3/auto-mpg-regression.tsv')
features2 = [('cylinders', q3.one_hot),
             ('displacement', q3.standard),
             ('horsepower', q3.standard),
             ('weight', q3.standard),
             ('acceleration', q3.standard),
             ('origin', q3.one_hot)]
X, y = q3.auto_data_and_values(all_data, features2)
mpg = y.flatten()
median = float(np.median(mpg))
y_bin = (mpg >= median).astype(int)

n = X.shape[1]
np.random.seed(0)
idx = np.arange(n)
np.random.shuffle(idx)
X = X[:, idx]
y_bin = y_bin[idx]
folds = np.array_split(np.arange(n), 10)


def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for j in range(X_test.shape[1]):
        x = X_test[:, j:j+1]
        dists = np.linalg.norm(X_train - x, axis=0)
        nn_idx = np.argpartition(dists, k)[:k]
        vote = np.sum(y_train[nn_idx])
        preds.append(1 if vote >= (k/2) else 0)
    return np.array(preds)


def metrics(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    tn = np.sum((y_true==0)&(y_pred==0))
    acc = (tp+tn)/len(y_true)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return acc, f1

Ks = [3,5,7]
results = {k: {'acc': [], 'f1': []} for k in Ks}

for fold_i in range(10):
    test_idx = folds[fold_i]
    train_idx = np.concatenate([folds[j] for j in range(10) if j!=fold_i])
    Xtr, Xte = X[:, train_idx], X[:, test_idx]
    ytr, yte = y_bin[train_idx], y_bin[test_idx]
    for k in Ks:
        yhat = knn_predict(Xtr, ytr, Xte, k)
        acc, f1 = metrics(yte, yhat)
        results[k]['acc'].append(acc)
        results[k]['f1'].append(f1)

for k in Ks:
    acc_mean = np.mean(results[k]['acc'])
    acc_std = np.std(results[k]['acc'])
    f1_mean = np.mean(results[k]['f1'])
    f1_std = np.std(results[k]['f1'])
    print(f"K={k}: Accuracy {acc_mean:.3f} +/- {acc_std:.3f}, F1 {f1_mean:.3f} +/- {f1_std:.3f}")
print(f"Threshold (median mpg) = {median:.3f}")
