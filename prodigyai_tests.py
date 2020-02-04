import os
import numpy as np

try:
    get_ipython()
    check_if_ipython = True
    os.chdir("timeseriesAI")
except Exception as e:
    print("Not using IPython")

from fastai_timeseries import *
print('pytorch:', torch.__version__)
print('fastai :', fastai.__version__)

import torch
import torch.nn as nn


class ROCKET(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels=10000, kss=[7, 9, 11]):
        '''
        ROCKET is a GPU Pytorch implementation of the ROCKET methods generate_kernels 
        and apply_kernels that can be used  with univariate and multivariate time series.
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS, 
        make sure you transform the 2d to 3d by adding unsqueeze(1).
        c_in: number of channels or features. For univariate c_in is 1.
        seq_len: sequence length
        '''
        super().__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0,
                                            np.log2((seq_len - 1) // (ks - 1)))
            padding = int(
                (ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, c_in, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in,
                              1,
                              ks,
                              padding=2 * padding,
                              dilation=int(dilation),
                              bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss

    def forward(self, x):
        for i in range(self.n_kernels):
            out = self.convs[i](x)
            _max = out.max(dim=-1).values
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            cat = torch.cat((_max, _ppv), dim=-1)
            output = cat if i == 0 else torch.cat((output, cat), dim=-1)
        return output


# ### SMALL DATASET with ridge regression

# # get the data
# X_train, y_train, X_valid, y_valid = get_UCR_data('OliveOil')
# seq_len = X_train.shape[-1]
# X_train = X_train[:, 0].astype(np.float64)
# X_valid = X_valid[:, 0].astype(np.float64)
# X_train.shape, X_valid.shape

# #normalize
# X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / (
#     X_train.std(axis=1, keepdims=True) + 1e-8)
# X_valid = (X_valid - X_valid.mean(axis=1, keepdims=True)) / (
#     X_valid.std(axis=1, keepdims=True) + 1e-8)
# X_train.mean(axis=1, keepdims=True).shape

# # generate kernels

# kss = np.array([7, 9, 11])
# kernels = generate_kernels(seq_len, 10000, kss=kss)

# # apply kernels to data
# X_train_tfm = apply_kernels(X_train, kernels)
# X_valid_tfm = apply_kernels(X_valid, kernels)

# # ridge regressor
# from sklearn.linear_model import RidgeClassifierCV
# classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7), normalize=True)
# classifier.fit(X_train_tfm, y_train)
# classifier.score(X_valid_tfm, y_valid)

# ### LARGE DATASET with logreg

# # Generate Features
# X_train, y_train, X_valid, y_valid = get_UCR_data('HandMovementDirection')
# _, features, seq_len = X_train.shape
# X_train = (X_train - X_train.mean(axis=(1, 2), keepdims=True)) / (
#     X_train.std(axis=(1, 2), keepdims=True) + 1e-8)
# X_valid = (X_valid - X_valid.mean(axis=(1, 2), keepdims=True)) / (
#     X_valid.std(axis=(1, 2), keepdims=True) + 1e-8)
# X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
# X_valid = torch.tensor(X_valid, dtype=torch.float32, device=device)
# print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# n_kernels = 10_000
# kss = [7, 9, 11]
# model = ROCKET(features, seq_len, n_kernels=n_kernels, kss=kss).to(device)

# X_train_tfm = model(X_train)
# X_valid_tfm = model(X_valid)

# # Ridge Regressor large dataset
# from sklearn.linear_model import RidgeClassifierCV
# ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17), normalize=True)
# ridge.fit(X_train_tfm, y_train)
# print('alpha: {:.2E}  train: {:.5f}  valid: {:.5f}'.format(
#     ridge.alpha_, ridge.score(X_train_tfm, y_train),
#     ridge.score(X_valid_tfm, y_valid)))

# # Log reg large dataset
# eps = 1e-6
# Cs = np.logspace(-5, 5, 11)
# from sklearn.linear_model import LogisticRegression
# best_loss = np.inf
# for i, C in enumerate(Cs):
#     f_mean = X_train_tfm.mean(axis=0, keepdims=True)
#     f_std = X_train_tfm.std(
#         axis=0, keepdims=True) + eps  # epsilon to avoid dividing by 0
#     X_train_tfm2 = (X_train_tfm - f_mean) / f_std
#     X_valid_tfm2 = (X_valid_tfm - f_mean) / f_std
#     classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
#     classifier.fit(X_train_tfm2, y_train)
#     probas = classifier.predict_proba(X_train_tfm2)
#     loss = nn.CrossEntropyLoss()(torch.tensor(probas),
#                                  torch.tensor(y_train)).item()
#     train_score = classifier.score(X_train_tfm2, y_train)
#     val_score = classifier.score(X_valid_tfm2, y_valid)
#     if loss < best_loss:
#         best_eps = eps
#         best_C = C
#         best_loss = loss
#         best_train_score = train_score
#         best_val_score = val_score
#     print(
#         '{:2} eps: {:.2E}  C: {:.2E}  loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'
#         .format(i, eps, C, loss, train_score, val_score))
# print('\nBest result:')
# print(
#     'eps: {:.2E}  C: {:.2E}  train_loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'
#     .format(best_eps, best_C, best_loss, best_train_score, best_val_score))

# n_tests = 10
# epss = np.logspace(-8, 0, 9)
# Cs = np.logspace(-5, 5, 11)

# from sklearn.linear_model import LogisticRegression
# best_loss = np.inf
# for i in range(n_tests):
#     eps = np.random.choice(epss)
#     C = np.random.choice(Cs)
#     f_mean = X_train_tfm.mean(axis=0, keepdims=True)
#     f_std = X_train_tfm.std(axis=0, keepdims=True) + eps  # epsilon
#     X_train_tfm2 = (X_train_tfm - f_mean) / f_std
#     X_valid_tfm2 = (X_valid_tfm - f_mean) / f_std
#     classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
#     classifier.fit(X_train_tfm2, y_train)
#     probas = classifier.predict_proba(X_train_tfm2)
#     loss = nn.CrossEntropyLoss()(torch.tensor(probas),
#                                  torch.tensor(y_train)).item()
#     train_score = classifier.score(X_train_tfm2, y_train)
#     val_score = classifier.score(X_valid_tfm2, y_valid)
#     if loss < best_loss:
#         best_eps = eps
#         best_C = C
#         best_loss = loss
#         best_train_score = train_score
#         best_val_score = val_score
#     print(
#         '{:2}  eps: {:.2E}  C: {:.2E}  loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'
#         .format(i, eps, C, loss, train_score, val_score))
# print('\nBest result:')
# print(
#     'eps: {:.2E}  C: {:.2E}  train_loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'
#     .format(best_eps, best_C, best_loss, best_train_score, best_val_score))
# #  bash local_run_script.sh timeseriesAI/prodigyai_tests.py temp-instance-gpu 1 one_model without_shutdown

# # run on a bunch of other datasets
# iterations = 5
# datasets = sorted(get_UCR_multivariate_list())
# n_kernels = 10_000
# kss = [7, 9, 11]

# ds_ = []
# ds_nl_ = []
# ds_di_ = []
# ds_type_ = []
# iters_ = []
# means_ = []
# stds_ = []
# times_ = []
# alphas_ = []

# datasets = listify(datasets)
# for dsid in datasets:
#     try:
#         print(f'\nProcessing {dsid} dataset...\n')
#         if dsid in get_UCR_univariate_list(): scale_type = None
#         else: scale_type = 'standardize'
#         data = create_UCR_databunch(dsid,
#                                     bs=2048,
#                                     scale_type=scale_type,
#                                     verbose=True)
#         X_train = data.train_ds.x.items.astype(np.float32)
#         X_valid = data.valid_ds.x.items.astype(np.float32)
#         y_train = data.train_ds.y.items.astype(np.int64)
#         y_valid = data.valid_ds.y.items.astype(np.int64)
#     except:
#         ds_nl_.append(dsid)
#         print(f'\n...{dsid} cannot be loaded\n')
#         continue
#     if np.isinf(X_train).sum() + np.isnan(X_train).sum() > 0:
#         ds_di_.append(dsid)
#         print(f'...{dsid} contains inf or nan\n')
#         continue

#     if data.features == 1: ds_type_.append('univariate')
#     else: ds_type_.append('multivariate')
#     score_ = []
#     elapsed_times_ = []
#     for i in range(iterations):
#         start_time = time.time()
#         model = ROCKET(data.features,
#                        data.seq_len,
#                        n_kernels=n_kernels,
#                        kss=kss).to(device)
#         for j, (xb, yb) in enumerate(
#                 data.train_dl.new(shuffle=False, drop_last=False)):
#             with torch.no_grad():
#                 out = model(xb)
#             X_train_tfm = out if j == 0 else torch.cat((X_train_tfm, out))
#         for j, (xb, yb) in enumerate(
#                 data.valid_dl.new(shuffle=False, drop_last=False)):
#             with torch.no_grad():
#                 out = model(xb)
#             X_valid_tfm = out if j == 0 else torch.cat((X_valid_tfm, out))
#         classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7),
#                                        normalize=True)
#         classifier.fit(X_train_tfm, y_train)
#         score = classifier.score(X_valid_tfm, y_valid)
#         score_.append(score)
#         elapsed_time = time.time() - start_time
#         elapsed_times_.append(elapsed_time)
#         alphas_.append(classifier.alpha_)
#         print('   {:2} - score: {:.5f}  alpha: {:.2E}  time(s): {:4.0f}'.\
#               format(i + 1, score, classifier.alpha_, elapsed_time))
#     ds_.append(dsid)
#     iters_.append(iterations)
#     means_.append('{:.6f}'.format(np.mean(score_)))
#     stds_.append('{:.6f}'.format(np.std(score_)))
#     times_.append('{:.0f}'.format(np.mean(elapsed_times_)))
#     if len(datasets) > 1: clear_output()
#     df = pd.DataFrame(np.stack(
#         (ds_, ds_type_, iters_, means_, stds_, times_)).T,
#                       columns=[
#                           'dataset', 'type', 'iterations', 'mean_accuracy',
#                           'std_accuracy', 'time(s)'
#                       ])
#     pd.set_option('display.max_columns', 999)
#     pd.set_option('display.max_rows', 999)
#     display(df)
#     pd.set_option('display.max_columns', 20)
#     pd.set_option('display.max_rows', 60)
#     if ds_nl_ != []: print('\n(*) datasets not loaded      :', ds_nl_)
#     if ds_di_ != []: print('(*) datasets with data issues:', ds_di_)

# # Integration with fastai
# # These are the steps to use ROCKET with a classifier in fastai:

# # Normalize the data 'per sample' (if not previously normalized)
# # Calculate features
# # Normalize calculated features 'per feature' (you'll get 20k means and stds)
# # Create the databunch passing the normalized calculated features as 3d tensor/arrays to a TSList.

# dsid = 'OliveOil'
# bs = 30
# eps = 1e-6

# # extract data
# X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)

# # normalize data 'per sample'
# X_train = (X_train - X_train.mean(axis=(1, 2), keepdims=True)) / (
#     X_train.std(axis=(1, 2), keepdims=True) + eps)
# X_valid = (X_valid - X_valid.mean(axis=(1, 2), keepdims=True)) / (
#     X_valid.std(axis=(1, 2), keepdims=True) + eps)

# # calculate 20k features
# _, features, seq_len = X_train.shape
# model = ROCKET(features, seq_len, n_kernels=10000, kss=[7, 9, 11]).to(device)
# X_train_tfm = model(torch.tensor(X_train, device=device)).unsqueeze(1)
# X_valid_tfm = model(torch.tensor(X_valid, device=device)).unsqueeze(1)

# # normalize 'per feature'
# f_mean = X_train_tfm.mean(dim=0, keepdims=True)
# f_std = X_train_tfm.std(dim=0, keepdims=True) + eps
# X_train_tfm_norm = (X_train_tfm - f_mean) / f_std
# X_valid_tfm_norm = (X_valid_tfm - f_mean) / f_std

# # create databunch
# data = (ItemLists('.', TSList(X_train_tfm_norm),
#                   TSList(X_valid_tfm_norm)).label_from_lists(
#                       y_train,
#                       y_valid).databunch(bs=min(bs, len(X_train)),
#                                          val_bs=min(bs * 2, len(X_valid))))

# # data.show_batch()

# # In this case we'll use a very simple: a Logistic Regression with
# # 20k input features and 2 classes in this case.
# def init(layer):
#     if isinstance(layer, nn.Linear):
#         nn.init.constant_(layer.weight.data, 0.)
#         nn.init.constant_(layer.bias.data, 0.)

# model = nn.Sequential(nn.Linear(20_000, data.c))
# model.apply(init)
# learn = Learner(data, model, metrics=accuracy)
# learn.save('stage-0')

# learn.lr_find()
# # learn.recorder.plot()

# learn.load('stage-0')
# learn.fit_one_cycle(10, max_lr=3e-3, wd=1e2)
# learn.recorder.plot_losses()
# learn.recorder.plot_metrics()

### with boosted trees (XG BOOST)

dsid = 'OliveOil'
eps = 1e-6

# extract data
X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)

# normalize data 'per sample'
X_train = (X_train - X_train.mean(axis=(1, 2), keepdims=True)) / (
    X_train.std(axis=(1, 2), keepdims=True) + eps)
X_valid = (X_valid - X_valid.mean(axis=(1, 2), keepdims=True)) / (
    X_valid.std(axis=(1, 2), keepdims=True) + eps)

# calculate 20k features
_, features, seq_len = X_train.shape
model = ROCKET(features, seq_len, n_kernels=10000, kss=[7, 9, 11]).to(device)
X_train_tfm = model(torch.tensor(X_train, device=device))
X_valid_tfm = model(torch.tensor(X_valid, device=device))

# normalize 'per feature'
f_mean = X_train_tfm.mean(dim=0, keepdims=True)
f_std = X_train_tfm.std(dim=0, keepdims=True) + eps
X_train_tfm_norm = (X_train_tfm - f_mean) / f_std
X_valid_tfm_norm = (X_valid_tfm - f_mean) / f_std

import xgboost as xgb
classifier = xgb.XGBClassifier(max_depth=3,
                               learning_rate=0.1,
                               n_estimators=100,
                               verbosity=1,
                               objective='binary:logistic',
                               booster='gbtree',
                               tree_method='auto',
                               n_jobs=1,
                               gpu_id=0,
                               gamma=0,
                               min_child_weight=1,
                               max_delta_step=0,
                               subsample=.5,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               colsample_bynode=1,
                               reg_alpha=0,
                               reg_lambda=1,
                               scale_pos_weight=1,
                               base_score=0.5,
                               random_state=0,
                               missing=None)

classifier.fit(X_train_tfm, y_train)
preds = classifier.predict(X_valid_tfm)
print((preds == y_valid).mean())

print("finished")