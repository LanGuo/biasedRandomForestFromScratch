#!/usr/bin/env python
'''
Random forest and biased random forest classifiers.
Refs:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier.feature_importances_
Bader-El-Den M. et al.(2019) Biased Random Forest For Dealing With the Class Imbalance Problem. DOI: 10.1109/TNNLS.2018.2878400
'''

import numpy as np
import multiprocessing as mp
from scipy.spatial import distance
from collections import Counter
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    '''
    Class for random forest classifier.
    '''
    def __init__(self,
                 n_estimators=100,
                 bootstrap=True,
                 max_sample_frac=None,
                 max_depth=None,
                 min_samples_leaf=2,
                 max_features=None,
                 n_processes=3,
                 random_seed=0):
        '''
        Params:
            n_estimators: int, default=100
                The number of trees in the forest.
            bootstrap: bool, default=True
                Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            max_sample_frac: float in interval `(0, 1)`, default=None
                If bootstrap is True, the fraction of total samples to draw from X to train each base estimator.
                If None (default), then draw all the samples.
            max_depth: int, default=None
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            min_samples_split: int, default=2
                The minimum number of samples required to split an internal node.
            max_features: int, default=None
                The number of features to consider when looking for the best split.
                If None, then all features will be considered.
            random_seed: int, default=0
                Controls the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True)
            n_processes: int, default=3
                Number of processes to run in parallel and speed up fitting of trees.
        '''
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        if max_sample_frac:
            assert (max_sample_frac > 0) & (max_sample_frac < 1), "max_sample_frac must be in the interval of (0,1)"
        self.max_sample_frac = max_sample_frac
        self.max_depth = (np.iinfo(np.int32).max if max_depth is None else max_depth)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.n_processes = n_processes
        self.estimators_ = []

    def fit(self, X, y):
        """Fit decision tree classifier."""
        np.random.seed(self.random_seed)
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._grow_forest(X, y)

    def predict_proba(self, X):
        '''
        Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.
        '''
        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        # sum all predicted probability from all trees
        for e in self.estimators_:
            proba = e.predict_proba(X)
            all_proba += proba
        all_proba /= len(self.estimators_)
        return all_proba

    def predict(self, X):
        '''
        Predict class label for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        '''
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def _bootstrap_sample(self, X, y):
        '''Bootstrap subsample'''
        np.random.seed(self.random_seed)
        bootstrap = self.bootstrap
        n_samples = X.shape[0]
        if not self.max_sample_frac:
            n_samples_bootstrap = n_samples
        else:
            n_samples_bootstrap = int(n_samples * self.max_sample_frac)
        idxs = np.random.randint(0, n_samples, n_samples_bootstrap)
        return X[idxs], y[idxs]

    def _fit_trees(self, tree, X, y):
        '''Subsample (if bootstrap) X and fit a tree.'''
        if self.bootstrap:
            X, y = self._bootstrap_sample(X, y)
        tree.fit(X, y)
        return tree

    def _grow_forest(self, X, y):
        '''Fit a random forest of decision trees'''
        trees = self._init_trees()
        if self.n_processes > 0:
            # parallel fitting trees
            pool = mp.Pool(processes=self.n_processes)
            calls = [pool.apply_async(self._fit_trees, args=(tree,
                        X, y)) for tree in trees]
            trees = [c.get() for c in calls]
            self.estimators_.extend(trees)
        else:
            # serial fitting trees
            for tree in trees:
                tree = self._fit_trees(tree, X, y)
                self.estimators_.append(tree)


    def _init_trees(self):
        trees = [DecisionTreeClassifier(max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_seed=self.random_seed,
                    n_processes=0)
                for i in range(self.n_estimators)]
        return trees


class BiasedRandomForestClassifier():
    '''
    Class for biased random forest classifier.
    '''
    def __init__(self,
                 critical_ratio=0.5,
                 k_nearest=10,
                 dist_metric='euclidean',
                 n_estimators=100,
                 bootstrap=True,
                 max_sample_frac=None,
                 max_depth=None,
                 min_samples_leaf=2,
                 max_features=None,
                 n_processes=3,
                 random_seed=0):
        '''
        Params:
            n_estimators: int, default=100
                The number of trees in the forest.
            bootstrap: bool, default=True
                Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            max_sample_frac: float in interval `(0, 1)`, default=None
                If bootstrap is True, the fraction of total samples to draw from X to train each base estimator.
                If None (default), then draw all the samples.
            max_depth: int, default=None
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            min_samples_split: int, default=2
                The minimum number of samples required to split an internal node.
            max_features: int, default=None
                The number of features to consider when looking for the best split.
                If None, then all features will be considered.
            random_seed: int, default=0
                Controls the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True)
            n_processes: int, default=3
                Number of processes to run in parallel and speed up fitting of trees.
            critical_ratio : float, default=0.5
                The proportion of random trees trained on the critical set, (1-critical_ratio) is the proportion of random trees train on original training set.
            k_nearest : int, default=10
                Number of nearest neighbors of each minority sample to extract from nonminority samples when building critical set.
            dist_metric : string, default='euclidean'
                The metric used for calculating distance (see scipy.spatial.distance.cdist for possible options).
        '''
        self.critical_ratio = critical_ratio
        if not (self.critical_ratio >= 0) and (self.critical_ratio <= 1):
            raise ValueError("critical_ratio must be in the range [0,1].")
        self.k_nearest = k_nearest
        self.dist_metric = dist_metric
        self.bootstrap = bootstrap
        self.max_sample_frac = max_sample_frac
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_processes = n_processes
        self.random_seed = random_seed
        self.n_estimators_unbiased = int(n_estimators * (1 - critical_ratio))
        self.rf_unbiased = RandomForestClassifier(self.n_estimators_unbiased,
                                                 bootstrap=bootstrap,
                                                 max_sample_frac=max_sample_frac,
                                                 max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 max_features=max_features,
                                                 n_processes=n_processes,
                                                 random_seed=random_seed)
        self.n_estimators_biased = n_estimators - self.n_estimators_unbiased
        self.rf_biased = RandomForestClassifier(self.n_estimators_biased,
                                                 bootstrap=bootstrap,
                                                 max_sample_frac=max_sample_frac,
                                                 max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 max_features=max_features,
                                                 n_processes=n_processes,
                                                 random_seed=random_seed)

    def fit(self, X, y):
        '''Fit a biased random forest classifier'''
        np.random.seed(self.random_seed)
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        critical_X, critical_y = self._build_critical_set(X, y)
        self.rf_unbiased.fit(X, y)
        self.rf_biased.fit(critical_X, critical_y)

    def predict_proba(self, X):
        '''
        Predict class probabilities for X as a weighted average predicted probability from both rfs
        '''
        # since the predicted class probabilities of an input sample are computed as
        # the mean predicted class probabilities of the trees in the forest,
        # each forest's contribution is weighted by the number of trees in each forest.
        proba_unbiased = self.rf_unbiased.predict_proba(X)
        proba_biased = self.rf_biased.predict_proba(X)
        all_proba = (self.n_estimators_unbiased * proba_unbiased + self.n_estimators_biased * proba_biased) \
                    / (self.n_estimators_unbiased + self.n_estimators_biased)
        return all_proba

    def predict(self, X):
        '''
        Predict class label for X as the one with highest mean probability
        estimate across the forests.
        '''
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def _build_critical_set(self, X, y):
        '''Build a critical set that includes all minority samples and their k-nearest majority neighbors'''
        (X_majority_neighbors, y_majority_neighbors), (X_minority, y_minority) = \
            self._nearest_neighbor_minority(X, y)
        # print(f'Minority samples: {X_minority.shape[0]}; Majority neighbors: {X_majority_neighbors.shape[0]}')
        critical_X = np.concatenate((X_minority, X_majority_neighbors), axis=0)
        critical_y = np.concatenate((y_minority, y_majority_neighbors), axis=0)
        # print(critical_X.shape, critical_y.shape)
        return critical_X, critical_y

    def _nearest_neighbor_minority(self, X, y):
        '''
        For each data point in the minority class,
        find its k nearest neighbors based on dist_metric from the non-minority classes.
        Params:
            X: ndarray, the data with samples in rows and features in columns.
            y: array, target label, should have length as number of samples in X.
        Returns:
            X_majority_neighbors: ndarray, data of samples in non-minority classes that are nearest neighbors to minority samples.
            X_minority: ndarray, data of samples in the minority class.
        '''
        k = self.k_nearest
        minority_class = sorted(Counter(y), reverse=True)[0]
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_nonminority = X[y != minority_class]
        y_nonminority = y[y != minority_class]
        dist = distance.cdist(self._normalize_features(X_minority),
                              self._normalize_features(X_nonminority),
                              self.dist_metric)
        idx_sorted = np.argsort(dist, axis=1)
        nearest_k_idxs = np.unique(idx_sorted[:, :k]) #only keep nearest neighbor once
        X_majority_neighbors = X_nonminority[nearest_k_idxs]
        y_majority_neighbors = y_nonminority[nearest_k_idxs]
        return (X_majority_neighbors, y_majority_neighbors), (X_minority, y_minority)

    def _normalize_features(self, X):
        '''
        Preprocessing for knn clustering by normalizing each feature in X to zero mean and unit variance.
        '''
        return (X - X.mean(axis=0)) / X.std(axis=0)