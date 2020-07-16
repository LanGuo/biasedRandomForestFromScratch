'''
Decision tree classifiers using Gini impurity index as decision metric.
Refs:
https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
'''
import numpy as np
import pandas as pd
import multiprocessing as mp


class DecisionTreeClassifier:
    '''
    Class for decision tree classifier.
    '''
    def __init__(self,
                 max_depth=None,
                 min_samples_leaf=2,
                 max_features=None,
                 random_seed=0,
                 n_processes=3
                 ):
        '''
        Params:
            max_depth: int, default=None
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            min_samples_split: int, default=2
                The minimum number of samples required to split an internal node.
            max_features: int, default=None
                The number of features to consider when looking for the best split.
                If None, then all features will be considered.
                Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
            random_seed: int, default=0
                Controls the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True)
            n_processes: int, default=3
                Number of processes to run in parallel and speed up finding best split.
        '''
        self.max_depth = (np.iinfo(np.int32).max if max_depth is None else max_depth)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.n_processes = n_processes
        self.depth = 0
        self.n_nodes = 0
        self.n_leafnodes = 0


    def fit(self, X, y):
        """Fit decision tree classifier."""
        np.random.seed(self.random_seed)
        # check X conforms to dtype supported and no missing values
        self._check_X(X)
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        if not self.max_features:
            self.max_features = self.n_features_
        elif self.max_features > self.n_features_:
            print("max_features larger than number of features available, using all features.")
        self.feature_types_ = ['binary' if len(np.unique(X[:, idx])) == 2 else 'continuous' for idx in range(self.n_features_)]
        self.tree_ = self._grow_tree(X, y)


    def _check_X(self, X):
        '''
        Check that X doesnot have missing value, since that is not supported.
        Check that X only contains features of type int or float,
        since only these types are supported.
        '''
        assert pd.notna(X).any(), "X can not have missing values."
        col_is_int = np.array([X[:, c].dtype.kind == 'i' for c in range(X.shape[1])])
        col_is_float = np.array([X[:, c].dtype.kind == 'f' for c in range(X.shape[1])])
        assert np.all(col_is_int | col_is_float), "All features of X needs to be converted to integer or float before fitting."


    def _check_y(self, y):
        '''
        Check that y is an 1-d array with integer or string values for class labels.
        Multi-output classification is not supported.
        '''
        assert y.ndim < 2, "Multioutput classification not supported."
        if y.dtype.kind == 'f':
            assert np.all(y == y.astype(int)), "y cannot be converted to integer class labels"
            y = y.astype(int)
        assert (np.issubdtype(y.dtype, np.str) or y.dtype.kind == 'i'), \
            "y must be of type integer or string"


    def _gini(self, y):
        """
        Compute Gini impurity of a non-empty node.
        Gini impurity is defined as Σ p(1-p) over all classes, with p the frequency of a
        class within the node. Since Σ p = 1, this is equivalent to 1 - Σ p^2.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in self.classes_)


    def _best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.
        Returns:
            best_idx: int. Index of the feature for best split, or None if no split is found.
            best_thr: float or int. Threshold to use for the split, or None if no split is found.
            rule: string. Spliting rule, 'equal' for binary features and 'smaller' for continuous features, None if no split is found.
        """
        # Need at least min_sample_leaf elements to split a node.
        m = y.size
        if m < self.min_samples_leaf:
            return None, None, None

        # Gini of current node.
        best_gini = self._gini(y)
        best_idx, best_thr, rule = None, None, None

        # parallel find splits in features
        if self.n_processes > 0:
            pool = mp.Pool(processes=self.n_processes)
        valid_splits = []
        idx_reserved = np.arange(self.n_features_)

        while (len(valid_splits) < 1) and (len(idx_reserved) > 0):
            if self.max_features and (self.max_features < self.n_features_):
                # test max_features number of features to find valid splits
                # Note: the search for a split does not stop until at least one valid partition of the node samples is found,
                # even if it requires to effectively inspect more than max_features features.
                try:
                    idx_to_split = np.random.choice(idx_reserved, self.max_features, replace=False)
                    idx_reserved = list(set(idx_reserved) - set(idx_to_split))
                except ValueError:
                    idx_to_split = idx_reserved
                    idx_reserved = list(set(idx_reserved) - set(idx_to_split))
            else:  # use all the features to look for split
                idx_to_split = idx_reserved
                idx_reserved = []
            if self.n_processes > 0: # parallel
                tests = [pool.apply_async(self._test_feature_split,
                        args=(idx, X, y))
                        for idx in idx_to_split]
                best_splits = [t.get() for t in tests] # each test result is (idx, best_gini, best_thr, rule)
            else: # serial
                best_splits = []
                for idx in idx_to_split:
                    split = self._test_feature_split(idx, X, y)
                best_splits.append(split)
            valid_splits = [s for s in best_splits if s[2]] # remove splits that do not have best_thr

        if len(valid_splits) > 0:
            # each split result is (idx, best_gini, best_thr, rule)
            valid_splits.sort(key=lambda x: x[1]) # sort by best_gini
            best_split_gini = valid_splits[0][1]
            if best_split_gini < best_gini: # only accept splits that decrease starting gini
                best_idx = valid_splits[0][0] # feature index with lowest best_gini
                best_thr = valid_splits[0][2]
                rule = valid_splits[0][3]
        return best_idx, best_thr, rule


    def _test_feature_split(self, idx, X, y):
        ''' Find the best split threshold in one feature '''
        best_gini = 1
        best_thr, rule = None, None
        m = y.size
        if self.feature_types_[idx] == 'binary':
            # test spliting by this binary feature
            threshold = np.unique(X[:, idx])[0]
            indices_left = (X[:, idx] == threshold)
            y_left = y[indices_left]
            y_right = y[~indices_left]
            gini_left = self._gini(y_left)
            gini_right = self._gini(y_right)

            # The Gini impurity of a split is the weighted average of the Gini
            # impurity of the children.
            gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m

            if gini < best_gini:
                best_gini = gini
                best_thr = threshold
                rule = 'equal'
            elif gini == best_gini:
                # when two splits have identical criterion, one split will be selected at random
                ridx = np.random.choice([0, 1])
                best_thr = [threshold, best_thr][ridx]
                rule = 'equal'

        elif self.feature_types_[idx] == 'continuous':
            # Sort data based on selected continuous feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            thresholds = np.array(thresholds)
            classes = np.array(classes)
            # test all thresholds in this continuous feature
            for i in range(1, m):  # possible split positions
                y_left = classes[:i]
                y_right = classes[i:]
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
                    rule = 'smaller'
                elif gini == best_gini:
                    # when multiple splits have identical criterion, one split will be selected at random
                    ridx = np.random.choice([0, 1])
                    best_thr = [(thresholds[i] + thresholds[i - 1]) / 2, best_thr][ridx]
                    rule = 'smaller'
        return (idx, best_gini, best_thr, rule)


    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node.
        # The predicted class is the one with largest population.
        # But if there's a tie, default to first class that has the biggest population
        num_samples_per_class = [np.sum(y == i) for i in self.classes_]
        predicted_class = self.classes_[np.argmax(num_samples_per_class)]
        class_prob = np.array(num_samples_per_class) / sum(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            class_prob=class_prob,
            depth=depth
        )

        self.n_nodes += 1
        # Split recursively unless maximum depth is reached or minimum samples per leaf is reached.
        if (depth < self.max_depth) and (X.shape[0] > self.min_samples_leaf):
            idx, thr, rule = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.rule = rule
                depth += 1
                node.left = self._grow_tree(X_left, y_left, depth)
                node.right = self._grow_tree(X_right, y_right, depth)
            else:
                self.n_leafnodes += 1
        else:
            self.n_leafnodes += 1
        if depth > self.depth:
            self.depth = depth
        return node


    def predict(self, X):
        '''
        Predict class label for X.
        The predicted class is the one with largest number of samples in a leaf.
        '''
        return np.array([self._predict(sample)[0] for sample in X])


    def predict_proba(self, X):
        '''
        Predict class probabilities for X.
        The predicted class probability is the fraction of samples in each class in a leaf.
        '''
        return np.array([self._predict(sample)[1] for sample in X])


    def _predict(self, inputs):
        """
        Go through the tree and predict the class of a single sample.
        Returns the predicted (majority) class and class probability for each class.
        """
        node = self.tree_
        while node.left:
            if node.rule == 'equal':
                if inputs[node.feature_index] == node.threshold:
                    node = node.left
                else:
                    node = node.right
            elif node.rule == 'smaller':
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.predicted_class, node.class_prob


class Node:
    '''
    Class to store properties of a node on the decision tree classifier.
    '''
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class, class_prob, depth):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.class_prob = class_prob
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.rule = None
        self.depth = 0