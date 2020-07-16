'''
Utility functions for spliting and subsampling data.
'''

import numpy as np


def df_safe_fillna(df, label_col, method='mode'):
	'''
	Fill na in dataframe, handles numeric and string values differently.
	Params:
		df: pandas dataframe.
		label_col: column that contains class label.
		method: string. {'mode', 'median'}.
	'''
	labels = df[label_col]
	df = df.drop([label_col], axis=1)
	if method == 'mode':
		df = df.fillna(df.mode().iloc[0])
	elif method == 'median':
		values = df.median(numeric_only=True)
		string_cols = [c for c in df.columns if df[c].dtype.kind == 'O']
		if len(string_cols):
			values[[string_cols]] = df[[string_cols]].mode()
		df = df.fillna(values)
	else:
		raise ValueError('Fill method not implemented.')
	df[label_col] = labels
	return df


def split_n_folds(X, y, n, fractions=None, shuffle=True, stratified=True):
	'''
	Split X (data) and y (target) into n folds with specified fractions.
	Option to stratify based on target (y).
	Params:
		X: ndarray, the data with samples in rows and features in columns.
		y: array, target label, should have length as number of samples in X.
		n: int, number of folds to split data and target into.
		fractions: array of float, fraction of samples to allocate into each fold, length of n, sums to 1.
		shuffle: bool, optional to shuffle the order of samples.
		stratified: bool, optional to stratify based on target (y).
	Returns:
		inds_each_fold: array of int, indices of rows in X for each fold.
		X_each_fold: ndarray, the data each fold.
		y_each_fold: array, taget labels each fold.
	'''
	if not fractions:
		fractions = np.repeat(1.0 / n, n)
	assert len(fractions) == n, "fractions should have length n."
	assert len(y) == X.shape[0], "Length of y should equal the number of rows in X."
	fractions = np.array(fractions)
	assert fractions.sum() == 1.0, "fractions should sum to 1."
	classes = np.unique(y)
	if stratified:
		folds_each_class = []
		for c in classes:
			idx_this_class = np.flatnonzero(y == c)
			if shuffle:
				np.random.shuffle(idx_this_class)
			nums_each_fold = np.floor(len(idx_this_class) * fractions).astype(int)
			# when number of samples not evenly divided by n, let last fold have the extra samples
			nums_each_fold[-1] += len(idx_this_class) - nums_each_fold.sum()
			fold_idxs_this_class = np.split(idx_this_class, np.cumsum(nums_each_fold)[:-1])
			folds_each_class.append(fold_idxs_this_class)
		inds_each_fold = [np.concatenate(fold) for fold in zip(*folds_each_class)]
		X_each_fold = [X[inds] for inds in inds_each_fold]
		y_each_fold = [y[inds] for inds in inds_each_fold]
	else:
		idx = np.arange(len(y))
		if shuffle:
			np.random.shuffle(idx)
		nums_each_fold = np.floor(len(y) * fractions).astype(int)
		# when number of samples not evenly divided by n, let last fold have the extra samples
		nums_each_fold[-1] += len(y) - nums_each_fold.sum()
		inds_each_fold = np.split(idx, np.cumsum(nums_each_fold)[:-1])
		X_each_fold = [X[inds] for inds in inds_each_fold]
		y_each_fold = [y[inds] for inds in inds_each_fold]
	return inds_each_fold, X_each_fold, y_each_fold