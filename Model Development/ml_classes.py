import copy

import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler

from util_classes import (Dataset, FeatureImportances, Model_list,
                          ModelOptions, Preprocess, SamplingMethods,
                          TrainValidationSplit, TunableParameters)


class ModelDevelopment:

    def __init__(self, data, params):
        """Initiates the class

        Args:
            data (pandas dataframe): the data that is used to train and evaluate the models
            params (class): an instance of the Parameters class with the development parameters
        """
        # store the data to be used by the class instance
        self.original_data = data
        # store the parameters to be used by the class instance
        self.params = params
        # define the preprocessing steps that need to be carried out
        self.preprocessing = Preprocess(params, self.params.STANDARDIZE_DATA, self.params.NORMALIZE_DATA,
                                        self.params.PCA_DATA, self.params.POLY_DATA, self.params.POWER_DATA)
        # define the sampling methods that might be used
        self.sampling_methods = SamplingMethods(params)
        # split the data into a training and test set
        self.dataset = Dataset(data.X, data.y, data.le,
                               params, self.sampling_methods)
        # define the hyperparameters that might be tuned
        self.tunable_params = TunableParameters()
        # define the models that are available for development
        self.available_models = ModelOptions()
        # check if the models are valid
        self.available_models.check_validity(self.params.MODELS)
        # define a list of models to be evaluated
        self.model_list = Model_list(
            self.params.MODELS, self.available_models, self.tunable_params, params)

    def pipeline(self):
        """A pipeline that performs optimization, feature selection, hyperparameter
        tuning, and evaluates the models in the list of models.
        """
        # check the validity of the sampling methods
        self.sampling_methods.check_validity(self.params.SAMPLING_METHOD)

        # perform the pipeline steps for each model independently
        for model in self.model_list.models:

            # step 1: optimize basic parameters
            print(f'Selecting the basic parameters for {model.name}')
            self.params.optimize(self.original_data, [model.name])

            # step 2: select features
            print(f'Selecting the best features for {model.name}')
            self.feature_selection(model)
            model.results.generate_fs_report(
                model.name, self.params.DATA_TYPE, self.params.PREDICTION_TYPE)

            # step 3: tune the hyperparameters
            print(f'Tuning the hyperparamters for {model.name}')
            self.tune_parameters_with_gridsearch(model)

            # step 4: evaluate model
            # if we are purely evaluating the model, perform cross-validation
            if self.params.TRAINING:

                # reset some parameters that were different for previous steps
                self.params.COMPUTE_CONFIDENCE_INTERVALS = True
                self.params.CV_FOLD = 5

                # evaluate the model in a cross-validation loop
                self.cross_validation(model)

                # generate the hyperparameter results and store them
                model.results.generate_hpt_report(
                    model.name, self.params.DATA_TYPE, self.params.PREDICTION_TYPE)
                # generate the confidence intervals and store them
                model.results.confidence_intervals_to_table(
                    model.name, self.params.DATA_TYPE, self.params.PREDICTION_TYPE)

            # if we are finalizing the model for production, train and evaluate on entire dataset
            else:
                # train and evaluate model on entire dataset
                self.finalize_model(model)
                # generate the results of hyperparameter tuning and store them
                model.results.generate_hpt_report(
                    model.name, self.params.DATA_TYPE, self.params.PREDICTION_TYPE)

        # regenerate the results to include all models in a single report
        if self.params.TRAINING:
            self.model_list.generate_hpt_report()
            self.model_list.confidence_intervals_to_table()
        else:
            self.model_list.generate_hpt_report()
            self.model_list.cv_or_tts_results_to_table('tts')

    def feature_selection(self, model):
        """Select the best features for a given model.

        Args:
            model (class instance): model for which features are selected
        """
        # define a class instance to store the results
        self.feature_importances = FeatureImportances(self.params)

        # ensure that we start with original dataset
        self.X_copy = copy.deepcopy(self.dataset.X_copy)
        self.y_copy = copy.deepcopy(self.dataset.y_copy)

        # compute feature importances by combining several feature selection techniques
        self.complete_feature_importance_computations(model)

        # obtain the averaged feature importances
        feature_importances = self.feature_importances.combined_importances

        # sort the features by importance
        feature_importances.sort_values(
            by='importance', ascending=False, inplace=True)

        best_accuracy = 0
        model.results.fs_thresholds = []
        model.results.fs_performance = []

        # go through a range of thresholds and evaluate which results in highest accuracy
        for threshold in range(5, len(self.columns), 5):

            # obtain the top threshold features
            features = feature_importances['feature'][0:threshold]

            # keep only those features in the data
            self.dataset.X = self.dataset.X[features]

            # perform cross-validation to evaluate performance
            self.cross_validation(model)

            # obtain the score
            balanced_accuracy = model.results.mean_balanced_accuracy

            # if the score is better than the currently best score
            if balanced_accuracy >= best_accuracy:
                # store that score as the new best score
                best_accuracy = balanced_accuracy
                # store the threshold as the new best threshold
                best_threshold = threshold

            # reset the data for the next iteration
            self.dataset.X = copy.deepcopy(self.X_copy)
            self.dataset.y = copy.deepcopy(self.y_copy)

            # store the threshold and score
            model.results.fs_thresholds.append(threshold)
            model.results.fs_performance.append(balanced_accuracy)

        # keep feature subset that led to the highest score
        selected_features = feature_importances['feature'][0:best_threshold]
        self.dataset.X = self.dataset.X[selected_features]
        self.dataset.X_TEST = self.dataset.X_TEST[selected_features]

    def complete_feature_importance_computations(self, model):
        """Combine several feature importance methods to perform feature selection
        """
        # obtain the column names
        self.columns = self.dataset.X.columns.to_list()

        # preprocess the dataset
        self.dataset.preprocess()

        # create dataframes from the data
        self.dataset.X = pd.DataFrame(self.dataset.X, columns=self.columns)
        self.dataset.X_TEST = pd.DataFrame(
            self.dataset.X_TEST, columns=self.columns)

        # compute ridge feature importances
        self.feature_importances.compute_ridge_feature_importances(
            self.dataset.X, self.dataset.y, self.columns, model.name)
        # compute random forest feature importances
        self.feature_importances.compute_rf_feature_importance(
            self.dataset.X, self.dataset.y, self.columns, model.name)
        # compute permutation feature importances
        self.feature_importances.compute_permutation_feature_importance(
            self.dataset.X, self.dataset.y, self.columns, model.name)
        # compute kbest feature importances
        self.feature_importances.compute_Kbest_feature_importance(
            self.dataset.X, self.dataset.y, self.columns, model.name)

        # combine the results of the previously performed feature importance methods
        self.feature_importances.combine_fi_methods(self.columns, model.name)

        # reset data for hp tuning
        self.dataset.X = copy.deepcopy(self.dataset.X_copy)
        self.dataset.X_TEST_copy = copy.deepcopy(self.dataset.X_TEST_copy)
        self.dataset.y = copy.deepcopy(self.dataset.y_copy)

    def tune_parameters_with_gridsearch(self, model, set_best_params=True):
        """Tune hyperparameters of a given model in a grid search

        Args:
            model (class instance): model of which the parameters are tuned
            set_best_params (bool, optional): Whether to set the params to the best params that were found. Defaults to True.

        Returns:
            dict: the best parameters that were found
        """
        # define the sampling method to be used, if any
        if self.params.SAMPLING_METHOD is not None:
            strategy_dict = {key: value for key, value in self.sampling_methods.__dict__.items(
            ) if not key.startswith('__') and not callable(key)}
            self.sampler = strategy_dict[self.params.SAMPLING_METHOD]
        else:
            self.sampler = None

        # create a pipeline
        pipeline = make_pipeline(
            StandardScaler(), self.sampler, model.predictor)

        print(f"Tuning parameters of {model.name}")

        # perform the grid search
        cv = StratifiedKFold(n_splits=3)
        clf = GridSearchCV(pipeline, model.tunable_params, scoring='balanced_accuracy',
                           cv=cv, verbose=10, n_jobs=-1)  # , n_iter = 200
        clf.fit(self.dataset.X, self.dataset.y)

        # add the best params that were found to the results
        best_params = clf.best_params_
        keys = [key.split('__')[1] for key in best_params.keys()]
        values = best_params.values()
        best_params = dict(zip(keys, values))
        model.results.best_params = best_params

        # set the hyperparameters of the model the best found parameter values
        if set_best_params:
            model.predictor.set_params(**best_params)

        return best_params

    def cross_validation(self, model):
        """Perform cross validation with given model.

        Args:
            model (class object): model to train cross-validation folds. Defaults to 10.
        """
        # define type of cv based on whether confidence intervals are computed or not
        if self.params.COMPUTE_CONFIDENCE_INTERVALS:
            kf = RepeatedStratifiedKFold(
                n_splits=self.params.CV_FOLDS, n_repeats=self.params.N_REPEATS, random_state=128)
        else:
            kf = StratifiedKFold(
                n_splits=self.params.CV_FOLDS, random_state=128)

        print(
            f'Performing {self.params.CV_FOLDS}-fold cross-validation...............')

        # evaluate the model in a cross-validation loop
        for fold, (train_index, validation_index) in enumerate(kf.split(self.dataset.X, self.dataset.y), 1):
            print(f'fold {fold}:')

            # split the data into a train and validation set
            X_train = self.dataset.X.iloc[train_index]
            y_train = self.dataset.y.iloc[train_index]
            X_validation = self.dataset.X.iloc[validation_index]
            y_validation = self.dataset.y.iloc[validation_index]

            # preprocess the split data
            split_data = TrainValidationSplit(
                self.params, X_train, X_validation, y_train, y_validation, self.sampling_methods)
            split_data.preprocess()

            # train the model on the training data
            model.train(split_data.X_train, split_data.y_train)

            # predict the validation set
            y_pred, y_proba = model.predict_validation(split_data)

            # compute scores for each performance metric and add the scores to the results
            model.results.add_scores(
                self.params, split_data.y_validation, y_pred, y_proba)
            # compute the confusion matrix and add it to the results
            model.results.add_confusion_matrix(split_data.y_validation, y_pred)

        # average the scores over all folds
        model.results.average_scores()
        # create a dictionary with the scores
        model.results.scores_to_dict()
        # compute the confidence intervals of the metrics
        model.results.compute_confidence_intervals()
        # print and save the scores
        model.results.show_scores()
        # plot and save the confusion matrix
        model.results.plot_confusion_matrix_validation(
            self.dataset.labels, self.params.DATA_TYPE, self.params.PREDICTION_TYPE)

        # possibly perform an error analysis
        if self.params.ERROR_ANALYSIS:
            model.error_analysis(split_data, self.dataset.X.columns.to_list())

    def finalize_model(self, model):
        """Train a model on the entire training set, evaluate it, and store it

        Args:
            model (class instance): model to train
        """
        # preprocess the entire dataset
        self.dataset.preprocess()

        # train the model and the training set
        model.train(self.dataset.X, self.dataset.y)

        # predict the test set
        y_pred, y_proba = model.predict_test(
            self.dataset.X_TEST, self.dataset.y_TEST)

        # add the scores to the results
        model.results.add_scores(
            self.params, self.dataset.y_TEST, y_pred, y_proba)

        # average the scores
        model.results.average_scores()

        # create a dictionary with the scores
        model.results.scores_to_dict()

        # show and save the results
        model.results.show_scores()

        # plot and save the confusion matrix
        model.results.plot_confusion_matrix_test(model.name, model.fitted_estimator, self.dataset.X_TEST, self.dataset.y_TEST,
                                                 self.params.labels, self.params.DATA_TYPE, self.params.PREDICTION_TYPE)

        # store the model for production
        model.save()

        # store the dataset
        self.dataset.save()

    def cross_validate_all_models(self):
        """Cross validate all models in the list of models (without any tuning, etc.)
        """
        # check validity of sampling methods
        self.sampling_methods.check_validity(self.params.SAMPLING_METHOD)
        # cross-validate each model in the list of models
        for model in self.model_list.models:
            self.cross_validation(model)

        # possibly compute confidence intervals and store the results
        if not self.params.COMPUTE_CONFIDENCE_INTERVALS:
            self.model_list.cv_or_tts_results_to_table('cv')
        else:
            self.model_list.confidence_intervals_to_table()

    def finalize_and_save_models(self):
        """Train all models in the list of models on the entire dataset and save them
        """
        # preprocess the entire dataset
        self.dataset.preprocess()

        # train each model in the list of models on the entire dataset
        for model in self.model_list.models:
            model.train(self.dataset.X, self.dataset.y)
            model.save()
            self.dataset.save()
