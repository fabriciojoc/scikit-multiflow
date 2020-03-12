import pandas as pd
import numpy as np

import warnings

from skmultiflow.data import DataStream


class RawDataStream(DataStream):
    """ Creates a raw stream from a data source.

    DataStream takes the whole data set containing the `X` (features) and `Y` (targets) or takes `X` and `Y` separately.
    For the first case `target_idx` and `n_targets` need to be provided, in the second case they are not needed.

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame (Default=None)
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    y: np.ndarray or pd.DataFrame, optional (Default=None)
        The targets' columns.

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.

    name: str, optional (default=None)
        A string to id the data.

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    Notes
    -----
    The stream object provides upon request a number of samples, in a way such that old samples cannot be accessed
    at a later time. This is done to correctly simulate the stream context.

    """

    def _load_X_y(self):

        self.y = pd.DataFrame(self.y)

        check_data_consistency(self.y, self.allow_nan)
        check_data_consistency(self.X, self.allow_nan)

        self.n_samples, self.n_features = self.X.shape
        self.feature_names = self.X.columns.values.tolist()
        self.target_names = self.y.columns.values.tolist()

        self.y = self.y.values
        self.X = self.X.values

        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} '
                                 'exceeds n_features {}'.format(self.cat_features_idx, self.n_features))
        self.n_num_features = self.n_features - self.n_cat_features

        if np.issubdtype(self.y.dtype, np.integer):
            self.task_type = self._CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
        else:
            self.task_type = self._REGRESSION

        self.target_values = self._get_target_values()

    def _load_data(self):

        check_data_consistency(self.data, self.allow_nan)

        rows, cols = self.data.shape
        self.n_samples = rows
        labels = self.data.columns.values.tolist()

        if (self.target_idx + self.n_targets) == cols or (self.target_idx + self.n_targets) == 0:
            # Take everything to the right of target_idx
            self.y = self.data.iloc[:, self.target_idx:].values
            self.target_names = self.data.iloc[:, self.target_idx:].columns.values.tolist()
        else:
            # Take only n_targets columns to the right of target_idx, use the rest as features
            self.y = self.data.iloc[:, self.target_idx:self.target_idx + self.n_targets].values
            self.target_names = labels[self.target_idx:self.target_idx + self.n_targets]

        self.X = self.data.drop(self.target_names, axis=1).values
        self.feature_names = self.data.drop(self.target_names, axis=1).columns.values.tolist()

        _, self.n_features = self.X.shape
        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} '
                                 'exceeds n_features {}'.format(self.cat_features_idx, self.n_features))
        self.n_num_features = self.n_features - self.n_cat_features

        if np.issubdtype(self.y.dtype, np.integer):
            self.task_type = self._CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
        else:
            self.task_type = self._REGRESSION

        self.target_values = self._get_target_values()


def check_data_consistency(raw_data_frame, allow_nan=False):
    """
    Check data consistency with respect to scikit-multiflow assumptions:

    * Only numeric data types are used.
    * Missing values are, in general, not supported.

    Parameters
    ----------
    raw_data_frame: pandas.DataFrame
        The data frame containing the data to check.

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    """
    # if (raw_data_frame.dtypes == 'object').values.any():
    #     # scikit-multiflow assumes that data is numeric
    #     raise ValueError('Non-numeric data found:\n {}'
    #                      'scikit-multiflow only supports numeric data.'.format(raw_data_frame.dtypes))

    if raw_data_frame.isnull().values.any():
        if not allow_nan:
            raise ValueError("NaN values found. Missing values are not fully supported.\n"
                             "You can deactivate this error via the 'allow_nan' option.")
        else:
            warnings.warn("NaN values found. Functionality is not guaranteed for some methods. Proceed with caution.",
                          UserWarning)
