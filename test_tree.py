#!/usr/bin/env python

'''
Test DecisionTreeClassifier implementation and compared to sklearn.
'''
import argparse
import numpy as np
import pandas as pd
import importlib
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.utils import Bunch
import decision_tree as dt
import sampling_utils as utils
import evaluation
importlib.reload(utils)
importlib.reload(dt)
importlib.reload(evaluation)


parser = argparse.ArgumentParser(description="Train and test a decision tree.")
parser.add_argument("--dataset", choices=["breast", "iris", "diabetes"], default="diabetes")
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--max_features", type=int, default=None)
parser.add_argument("--min_samples_leaf", type=int, default=2)
parser.add_argument("--n_processes", type=int, default=4)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--test_fraction", type=float, default=0.10)
parser.add_argument("--hide_details", dest="hide_details", action="store_true")
parser.set_defaults(hide_details=False)
parser.add_argument("--compare_sklearn", dest="compare_sklearn", action="store_true")
parser.set_defaults(compare_sklearn=True)
parser.add_argument("--cross_validation", type=int, default=10)
args = parser.parse_args()

# Load dataset.
if args.dataset == "breast":
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
elif args.dataset == "iris":
    dataset = load_iris()
    X, y = dataset.data, dataset.target
elif args.dataset == "diabetes":
    df = pd.read_csv("diabetes.csv")
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

# Fit decision tree on training set with cross validation.
if args.compare_sklearn:
    clf_sk = SklearnDecisionTreeClassifier(max_depth=args.max_depth,
                                        max_features=args.max_features,
                                        min_samples_leaf=args.min_samples_leaf,
                                        random_state=args.random_seed)

clf = dt.DecisionTreeClassifier(max_depth=args.max_depth,
                                max_features=args.max_features,
                                min_samples_leaf=args.min_samples_leaf,
                                random_seed=args.random_seed,
                                n_processes=args.n_processes)



# n-fold cross validation on training set
n_cv = args.cross_validation
inds_each_fold, X_each_fold, y_each_fold = utils.split_n_folds(X_train, y_train,
    n=n_cv, fractions=None, stratified=True)
for n in range(n_cv):
    mask = np.isin(np.arange(X_train.shape[0]), inds_each_fold[n], invert=True)
    X_fit = X_train[mask]
    y_fit = y_train[mask]
    clf.fit(X_fit, y_fit)
    X_ho = X_train[~mask]
    y_ho = y_train[~mask]
    y_pred = clf.predict(X_ho)
    y_score = clf.predict_proba(X_ho)
    precision_by_class, recall_by_class = evaluation.precision_recall_score_by_class(y_ho, y_pred)
    tpr_classes, fpr_classes, roc_auc_classes = evaluation.roc_curve(y_ho, y_score, classes=[0.0, 1.0])

# 4. Visualize.
if args.use_sklearn:
    export_graphviz(
        clf,
        out_file="tree.dot",
        feature_names=dataset.feature_names,
        class_names=dataset.target_names,
        rounded=True,
        filled=True,
    )
    print("Done. To convert to PNG, run: dot -Tpng tree.dot -o tree.png")
else:
    clf.debug(
        list(dataset.feature_names),
        list(dataset.target_names),
        not args.hide_details,
    )


plt.figure()
lw = 2
plot_class = 0
plt.plot(fpr_classes[plot_class], tpr_classes[plot_class], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[plot_class])

ax.plot(recall_classes[plot_class], precision_classes[plot_class], color='darkorange',
         lw=lw, label='Precision-recall curve (area = %0.2f)' % prc_auc[plot_class])