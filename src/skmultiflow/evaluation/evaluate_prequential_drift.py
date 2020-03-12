import os
import warnings
import re
from copy import deepcopy
from timeit import default_timer as timer
import scipy.sparse as sp
import numpy as np

from numpy import unique
from skmultiflow.data.base_stream import Stream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants


class EvaluatePrequentialDrift(StreamEvaluator):
    """ The prequential evaluation method or interleaved test-then-train method.

    An alternative to the traditional holdout evaluation, inherited from
    batch setting problems.

    The prequential evaluation is designed specifically for stream settings,
    in the sense that each sample serves two purposes, and that samples are
    analysed sequentially, in order of arrival, and become immediately
    inaccessible.

    This method consists of using each sample to test the model, which means
    to make a predictions, and then the same sample is used to train the model
    (partial fit). This way the model is always tested on samples that it
    hasn't seen yet.

    Parameters
    ----------
    n_wait: int (Default: 200)
        The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.
        Note that setting `n_wait` too small can significantly slow the evaluation process.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    pretrain_size: int (Default: 200)
        The number of samples to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['accuracy', 'kappa'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
        | **Classification**
        | 'accuracy'
        | 'kappa'
        | 'kappa_t'
        | 'kappa_m'
        | 'true_vs_predicted'
        | 'precision'
        | 'recall'
        | 'f1'
        | 'gmean'
        | **Multi-target Classification**
        | 'hamming_score'
        | 'hamming_loss'
        | 'exact_match'
        | 'j_index'
        | **Regression**
        | 'mean_square_error'
        | 'mean_absolute_error'
        | 'true_vs_predicted'
        | **Multi-target Regression**
        | 'average_mean_squared_error'
        | 'average_mean_absolute_error'
        | 'average_root_mean_square_error'
        | **Experimental** (no plot generated)
        | 'running_time'
        | 'model_size'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
        process.

    restart_stream: bool, optional (default: True)
        If True, the stream is restarted once the evaluation is complete.

    data_points_for_classification: bool(Default: False)
        If True, the visualization used is a cloud of data points (only works for classification) and default
        performance metrics are ignored. If specific metrics are required, then they *must* be explicitly set
        using the ``metrics`` attribute.

    Notes
    -----
    1. This evaluator can process a single learner to track its performance; or multiple learners  at a time, to
       compare different models on the same stream.

    2. The metric 'true_vs_predicted' is intended to be informative only. It corresponds to evaluations at a specific
       moment which might not represent the actual learner performance across all instances.

    3. The metrics `running_time` and `model_size ` are not plotted when the `show_plot` option is set. Only their
       current value is displayed at the bottom of the figure. However, their values over the evaluation are written
       into the resulting csv file if the `output_file` option is set.

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Set the model
    >>> ht = HoeffdingTreeClassifier()
    >>>
    >>> # Set the evaluator
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=10000,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    >>> # The second example demonstrates how to compare two models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.bayes import NaiveBayes
    >>> from skmultiflow.evaluation import EvaluateHoldout
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Set the models
    >>> ht = HoeffdingTreeClassifier()
    >>> nb = NaiveBayes()
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=10000,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])

    >>> # The third example demonstrates how to evaluate one model
    >>> # and visualize the predictions using data points.
    >>> # Note: You can not in this case compare multiple models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> # Set the model
    >>> ht = HoeffdingTreeClassifier()
    >>> # Set the evaluator
    >>> evaluator = EvaluatePrequential(max_samples=200,
    >>>                                 n_wait=1,
    >>>                                 pretrain_size=1,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy'],
    >>>                                 data_points_for_classification=True)
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    """

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False):

        super().__init__()
        self._method = 'prequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.drifts = []

        if not self.data_points_for_classification:
            if metrics is None:
                self.metrics = [constants.ACCURACY, constants.KAPPA]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        else:
            if metrics is None:
                self.metrics = [constants.DATA_POINTS]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                    self.metrics.append(constants.DATA_POINTS)
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def _init_evaluation(self, stream, model, model_names=None):
        # For now, it just suports a single evaluation
        if isinstance(model, list):
            if len(model) > 1:
                raise ValueError("EvaluatePrequentialDrift does not support multiple models yet.")
            else:
                self.n_models = 1
        else:
            self.n_models = 1
            if not hasattr(model, 'predict'):
                raise NotImplementedError('{} does not have a predict() method.'.format(model))

        self.model = model if isinstance(model, list) else [model]

        if isinstance(stream, Stream):
            self.stream = stream
        else:
            raise ValueError('{} is not a valid stream type.'.format(stream))

        if model_names is None:
            self.model_names = ['M{}'.format(i) for i in range(self.n_models)]
        else:
            if isinstance(model_names, list):
                if len(model_names) != self.n_models:
                    raise ValueError("Number of model names does not match the number of models.")
                else:
                    self.model_names = model_names
            else:
                raise ValueError("model_names must be a list.")

    def evaluate(self, stream, model,
                 drift_detection_method: BaseDriftDetector = ADWIN(0.001),
                 warning_detection_method: BaseDriftDetector = ADWIN(0.01),
                 model_names=None,
                 extractor=None,
                 scaler=None):
        """ Evaluates a model or set of models on samples from a stream.

        Parameters
        ----------
        drift_detection_method: BaseDriftDetector or None, optional (default=ADWIN(0.001))
            Drift Detection method.
        warning_detection_method: BaseDriftDetector or None, default(ADWIN(0.01))
            Warning Detection method.

        stream: Stream
            The stream from which to draw the samples.

        model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        extractor: feature extractor to be used. Must have a fit and transform method.
        If this is enabled, data stream must be composed of RAW data. Both classifiers
        and drift detector will be reseted when a drift occurs.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)
        # save base model and extractor (if it is enabled)
        self.base_model = deepcopy(model)
        if extractor is not None:
            self.base_extractor = deepcopy(extractor)
        if scaler is not None:
            self.base_scaler = deepcopy(scaler)

        self.drift_detector = drift_detection_method
        self.warning_detector = warning_detection_method

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation Drift')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        self.actual_max_samples = self.stream.n_remaining_samples()
        if self.actual_max_samples == -1 or self.actual_max_samples > self.max_samples:
            self.actual_max_samples = self.max_samples

        self.first_run = True
        if self.pretrain_size > 0:

            data, y = self.stream.next_sample(self.pretrain_size)

            # check if feature extractor is enabled
            if self.base_extractor is not None:
                print('Extracting feature from {} sample(s).'.format(self.pretrain_size))

                # extract features
                self.extractor = FeatureExtractor(self.base_extractor)
                self.extractor.fit(data)
                X = self.extractor.transform(data)

                # if a scaler is available, normalize items
                if self.base_scaler is not None:
                    self.scaler = Scaler(self.base_scaler)
                    self.scaler.fit(X)
                    X = self.scaler.transform(X)
            else:
                X = data

            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            self.first_run = False

        self.update_count = 0
        print('Evaluating...')
        while ((self.global_sample_count < self.actual_max_samples) & (
                self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:

                    # check if feature extractor is enabled
                    if self.base_extractor is not None:
                        X = self.extractor.transform(X)
                        # if a scaler is available, normalize items
                        if self.base_scaler is not None:
                            X = self.scaler.transform(X)

                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                            # Testing time
                            self.running_time_measurements[i].compute_testing_time_begin()
                            prediction[i].extend(self.model[i].predict(X))
                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.drift_detector.add_element(int(y[i]==prediction[j][i]))
                    self._check_progress(self.actual_max_samples)

                    # check if has change to change model
                    if self.drift_detector.detected_change():
                        # print("Change detected in {}.".format((self.global_sample_count)))
                        self.drifts.append(self.global_sample_count)
                        # get last window of data
                        data, y = self.stream.get_window(self.drift_detector.width)
                        # extract feature again if has extractor
                        if self.base_extractor is not None:
                            # extract features
                            self.extractor = FeatureExtractor(self.base_extractor)
                            self.extractor.fit(data)
                            X = self.extractor.transform(data)

                            # if a scaler is available, normalize items
                            if self.base_scaler is not None:
                                self.scaler = Scaler(self.base_scaler)
                                self.scaler.fit(X)
                                X = self.scaler.transform(X)
                        else:
                            X = data

                        # reset classifier
                        # for i in range(self.n_models):
                        #     self.model[i] = deepcopy()(self.base_model[i])
                        self.model= deepcopy(self.base_model)

                        # reset drift detector
                        self.drift_detector.reset()
                    # Train
                    if self.first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                                    self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        self.first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= self.max_samples) or
                            (self.global_sample_count / self.n_wait > self.update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()
                        self.update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        print("Drifts detected: {}".format(len(self.drifts)))
        print("Drift points: {}", self.drifts)

        return self.model

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification labels / target values for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task. Not used for regressors.

        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info


class FeatureExtractor:
    def __init__(self, extractor):
        # save models used
        self.models = []
        # base extractor
        self.base_extractor = extractor

    # fit models and return them
    def fit(self, X_data):
        # iterate over each column of X_data
        for i in range(X_data.shape[1]):
            # get train and test data
            train_data = X_data[:, i]
            # initialize model
            tfidf = deepcopy(self.base_extractor)  # clone(self.base_extractor)
            # train model
            tfidf.fit(train_data)
            # append
            self.models.append(tfidf)
        # return models list
        return self.models

    # transform passed data
    def transform(self, X_data):
        X = []
        # iterate over each column of X_data
        for i in range(X_data.shape[1]):
            # get data column
            train_data = X_data[:, i]
            # get model for this column
            model = self.models[i]
            # transform data
            X_i = model.transform(train_data).todense().tolist()
            # if first execution, save only features
            if len(X) == 0:
                X = X_i
            # concatenate existing features
            else:
                # iterate over X and X_i
                for xx, xt in zip(X, X_i):
                    # iterate over X_i and append to X
                    for i in xt:
                        # append i to xx
                        xx.append(i)
                # X = np.concatenate((X, X_i), axis=1)
        # return transformed data
        return np.array(X)

class Scaler():

    def __init__(self, scaler):
        self.scaler = deepcopy(scaler)

    def fit(self, X_data):
        # train scaler
        self.scaler.fit(X_data)
        # return trained scaler
        return self.scaler

    def transform(self,X_data):
        # transform X_data
        return self.scaler.transform(X_data)