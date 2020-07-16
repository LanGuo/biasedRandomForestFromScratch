'''
Test RandomForestClassifier implementation and compared to sklearn.
'''
import os
import argparse
import numpy as np
import pandas as pd
import importlib
from collections import Counter
from matplotlib import pyplot as plt
import random_forest as rf
import sampling_utils as utils
import evaluation
importlib.reload(utils)
importlib.reload(rf)
importlib.reload(evaluation)


parser = argparse.ArgumentParser(description="Train and test random forest and biased random forest classifiers.")
parser.add_argument("--critical_ratio", type=float, default=0.5)
parser.add_argument("--dist_metric", type=str, default='euclidean')
parser.add_argument("--k_nearest", type=int, default=10)
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--bootstrap", type=bool, default=True)
parser.add_argument("--max_sample_frac", type=float, default=None)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--max_features", type=int, default=None)
parser.add_argument("--min_samples_leaf", type=int, default=2)
parser.add_argument("--n_processes", type=int, default=4)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--test_fraction", type=float, default=0.10)
parser.add_argument("--cross_validation", type=int, default=10)
parser.add_argument("--output_dir", type=str, default='cv_output')
args = parser.parse_args()

# Load dataset.
df = pd.read_csv("diabetes.csv")
# Fill missing values if there are any
df = utils.df_safe_fillna(df, label_col='Outcome', method='median')
feature_names = df.columns[:-1]
target_name = df.columns[-1]
data = df.to_numpy()[:, :-1]
target = df.to_numpy()[:, -1]
X, y = data, target

# Train test split (stratified)
test_fraction = args.test_fraction
assert test_fraction < 1, "Test fraction should be a float less than 1."
[train_inds, test_inds], [X_train, X_test], [y_train, y_test] = utils.split_n_folds(X, y,
    n=2, fractions=[1 - test_fraction, test_fraction], stratified=True)

# Fit random forest and BRAF on training set with cross validation.
clf = rf.RandomForestClassifier(n_estimators=args.n_estimators,
                                bootstrap=args.bootstrap,
                                max_sample_frac=args.max_sample_frac,
                                max_depth=args.max_depth,
                                max_features=args.max_features,
                                min_samples_leaf=args.min_samples_leaf,
                                random_seed=args.random_seed,
                                n_processes=args.n_processes)

clf_braf = rf.BiasedRandomForestClassifier(critical_ratio=args.critical_ratio,
                                           k_nearest=args.k_nearest,
                                           dist_metric=args.dist_metric,
                                            n_estimators=args.n_estimators,
                                            bootstrap=args.bootstrap,
                                            max_sample_frac=args.max_sample_frac,
                                            max_depth=args.max_depth,
                                            max_features=args.max_features,
                                            min_samples_leaf=args.min_samples_leaf,
                                            random_seed=args.random_seed,
                                            n_processes=args.n_processes)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# n-fold cross validation on training set
n_cv = args.cross_validation
inds_each_fold, X_each_fold, y_each_fold = utils.split_n_folds(X_train, y_train,
    n=n_cv, fractions=None, stratified=True)
for n in range(n_cv):
    print(f'Cross validation fold {n+1}')
    mask = np.isin(np.arange(X_train.shape[0]), inds_each_fold[n], invert=True)
    X_fit = X_train[mask]
    y_fit = y_train[mask]
    clf.fit(X_fit, y_fit)
    clf_braf.fit(X_fit, y_fit)

    X_ho = X_train[~mask]
    y_ho = y_train[~mask]
    y_pred = clf.predict(X_ho)
    y_score = clf.predict_proba(X_ho)
    precision_by_class, recall_by_class = evaluation.precision_recall_score_by_class(y_ho, y_pred)
    tpr_classes, fpr_classes, roc_auc_classes = evaluation.roc_curve(y_ho, y_score, classes=[0.0, 1.0])
    precision_classes, recall_classes, prc_auc_classes = evaluation.prc_curve(y_ho, y_score, classes=[0.0, 1.0])
    print('----- Unbiased Random Forest -----')
    print(f'Params: n_estimators={args.n_estimators},\
            max_sample_frac={args.max_sample_frac},\
            max_depth={args.max_depth},\
            max_features={args.max_features},\
            min_samples_leaf={args.min_samples_leaf},\
            ')
    print(f'precision each class: {precision_by_class}, \
            recall each class: {recall_by_class}, \
            AUC ROC curve each class: {roc_auc_classes}, \
            AUC Precision-recall curve each class: {prc_auc_classes}')
    print('-'*40)

    y_pred = clf_braf.predict(X_ho)
    y_score = clf_braf.predict_proba(X_ho)
    precision_by_class, recall_by_class = evaluation.precision_recall_score_by_class(y_ho, y_pred)
    tpr_classes, fpr_classes, roc_auc_classes = evaluation.roc_curve(y_ho, y_score, classes=[0.0, 1.0])
    precision_classes, recall_classes, prc_auc_classes = evaluation.prc_curve(y_ho, y_score, classes=[0.0, 1.0])
    print('----- Biased Random Forest -----')
    print(f'Params: critical_ratio={args.critical_ratio},\
            k_nearest={args.k_nearest},\
            dist_metric={args.dist_metric},\
            n_estimators={args.n_estimators},\
            max_sample_frac={args.max_sample_frac},\
            max_depth={args.max_depth},\
            max_features={args.max_features},\
            min_samples_leaf={args.min_samples_leaf},\
            ')
    print(f'precision each class: {precision_by_class}, \
            recall each class: {recall_by_class}, \
            AUC ROC curve each class: {roc_auc_classes}, \
            AUC Precision-recall curve each class: {prc_auc_classes}')
    print('-'*40)


    # Plot BRAF ROC curve and PRC curve and save figures
    lw = 2
    color = 'darkorange'
    minority_class = sorted(Counter(y_fit), reverse=True)[0]
    fig = evaluation.plot_roc_curve(fpr_classes[minority_class], tpr_classes[minority_class], minority_class, color=color,
                              lw=lw, label=f'CV={n+1}, ROC AUC={roc_auc_classes[minority_class]:.3f}')
    plt.savefig(os.path.join(output_dir, f'BRAF_cv{n+1}_roc_curve_class{minority_class}.png'))

    fig = evaluation.plot_prc_curve(recall_classes[minority_class], precision_classes[minority_class], minority_class, color=color,
                              lw=lw, label=f'CV={n+1}, Precision-recall curve AUC={prc_auc_classes[minority_class]:.3f}')
    plt.savefig(os.path.join(output_dir, f'BRAF_cv{n+1}_prc_curve_class{minority_class}.png'))
