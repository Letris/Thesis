
#################################################
# parameter file for St. Anna project
################################################

import numpy as np
from sklearn.model_selection import ParameterGrid
from ml_classes import ModelDevelopment
from dataclasses import dataclass, field, InitVar
from typing import Any, List, Dict, Set
import sys
import json

@dataclass
class Parameters:
    """Class to hold the parameters for model development
    """

    ##############################################
    # set below parameters based on  experiment
    ##############################################

    # the dataset (Knee or Hip)
    DATA_TYPE : str = 'Hip'
    # what are we predicting (Diagnosis or Treatment)
    PREDICTION_TYPE : str = 'Treatment'
    # whether we are evaluating on validation or test set
    VALIDATION_OR_TEST = "test"
    # the labels that we are predicting
    labels = ['Conservative', 'Surgical']
    # which models are we evaluating?
    MODELS = ['knn', 'dt', 'rf', 'adaboost', 'svm', 'gb', 'nn']

    # do we compute confidence intervals (True or False) MUST ALWAYS BE FALSE WHEN CALLING PIPELINE
    COMPUTE_CONFIDENCE_INTERVALS : bool = False
    # the amount of cross-validation folds
    CV_FOLDS : int = 3
    # how often do we repeat cv
    N_REPEATS : int = 10
    # is it a multiclass problem? (True or False)
    MULTICLASS = True
    # how often do we repeat a single feature importance method
    N_ROUNDS = 5
    # top n features to plot for feature importance scores
    FI_TOP_N = 10
    # analyze the errors of the models or not
    ERROR_ANALYSIS : bool = False

    ##############################################################
    # preprocessing, evaluation strategy, and sampling options
    ##############################################################

    # choose one of both:
    # normalize the data or not
    NORMALIZE_DATA : bool = True
    # do we standardize the data (True or False)
    STANDARDIZE_DATA : bool = False

    # apply PCA to the data or not
    PCA_DATA : bool =  False
    # the amount of variance to be explained by PCA if you choose to apply PCA
    PCA_VARIANCE : float = 0.6

    # add polynomial features to the data or not
    POLY_DATA : bool = False
    # the exponent to raise the features to if you choose to add polynomial features
    AMT_POLY : Any = 2

    # power the data or not
    POWER_DATA : bool = False

    # the sampling method to use (one of 'under', 'over', 'smote', 'smotetomek', 'both')
    SAMPLING_METHOD : str = 'under'
    # the sampling strategy to apply
    UNDERSAMPLE_STRATEGY :  Any = 'majority'
    OVERSAMPLE_STRATEGY : Any = 'minority'

    # choose one of both:
    # use the One vs One or One vs Rest multiclass strategy
    OVO : bool = False
    OVR : bool = False

    CC : bool = False

    def __post_init__(self):
        """Initialise the grid for optimization of the parameters
        """
        self.optimization_options = {
            # variable ENTER WHAT YOU WANT
            'SAMPLING_METHOD' : [None, 'under', 'over', 'smote', 'smotetomek'],
            'UNDERSAMPLE_STRATEGY' : [None] + ['majority'],
            'OVERSAMPLE_STRATEGY' : [None] + ['minority'],
            'OVO' : [ False],
            'OVR' : [True, False],
            'CC' : [True, False],
            'POLY_DATA' : [False],
            'POWER_DATA' : [False],
            'AMT_POLY' : [None],
            'STANDARDIZE_DATA' : [True],
            'NORMALIZE_DATA' : [False],
            'PCA_DATA' : [False],
            # The amount of variance to be explained by PCA
            'PCA_VARIANCE' : [None],

            # the parameter below are always the same, do not change
            # the dataset (Knee or Hip)
            'DATA_TYPE' : [self.DATA_TYPE],
            # what are we predicting (Diagnosis or Treatment)
            'PREDICTION_TYPE' : [self.PREDICTION_TYPE],
            # which models are we evaluating?
            'MODELS' : [self.MODELS],
            # do we compute confidence intervals (True or False)
            'COMPUTE_CONFIDENCE_INTERVALS' : [self.COMPUTE_CONFIDENCE_INTERVALS],
            # do we standardize the data (True or False)
            'STANDARDIZE_DATA' : [self.STANDARDIZE_DATA],
            # the amount of cross-validation folds
            'CV_FOLDS' : [self.CV_FOLDS],
            # how often do we repeat cv
            'N_REPEATS' : [self.N_REPEATS],
            # is it a multiclass problem? (True or False)
            'MULTICLASS' : [self.MULTICLASS],
            # are we training or are we fitting the final models
            'TRAINING' : [self.VALIDATION_OR_TEST],
            # how often do we repeat a single feature importance method
            'N_ROUNDS' : [self.N_ROUNDS],
            'FI_TOP_N' : [self.FI_TOP_N]
            }

        # make a grid of the optimization options
        self.param_grid = list(ParameterGrid(self.optimization_options))

        # filter invalid combinations from the grid
        self.filter_param_grid()

    def filter_param_grid(self):
        """Filter invalid parameter combinations from the optimization grid
        """
        filtered_grid = []

        for params in self.param_grid:
            if params['SAMPLING_METHOD'] != 'under' and params['UNDERSAMPLE_STRATEGY'] != None:
                continue
            if params['SAMPLING_METHOD'] != 'over' and params['OVERSAMPLE_STRATEGY'] != None:
                continue
            if params['SAMPLING_METHOD'] == 'over' and params['OVERSAMPLE_STRATEGY'] == None:
                continue
            if params['SAMPLING_METHOD'] == 'under' and params['UNDERSAMPLE_STRATEGY'] == None:
                continue
            if params['PCA_VARIANCE'] != None:
                if params['PCA_DATA'] == False and params['PCA_VARIANCE'] > 0.4:
                    continue
            if params['STANDARDIZE_DATA'] == True and params['NORMALIZE_DATA'] == True:
                continue
            if params['OVO'] == True and params['OVR'] == True:
                continue
            if params['OVO'] == True and params['CC'] == False:
                continue
            if params['POLY_DATA'] == False and params['AMT_POLY'] != None:
                continue
            if params['POLY_DATA'] == True and params['AMT_POLY'] == None:
                continue

            filtered_grid.append(params)

        self.param_grid = filtered_grid

    def optimize(self, data, model=['rf']):
        """Find the best combination of parameter values

        Args:
            data (class instance): The data to use for optimization
            model (list, optional): Models for which to optimize parameter values. Defaults to ['rf'].

        Returns:
            dict: best combination of parameter values
        """

        # placeholders for best metric scores
        current_best_AUC = 0
        current_best_balanced_accuracy = 0

        # for each unique combination of parameters in the grid
        for i, params in enumerate(self.param_grid):

            print(f'Evaluating #{i} out of {len(self.param_grid)} parameter configurations')

            # set the parameters to the combination of parameters
            self.update(params)

            # retrieve the parameters in dictionary form
            params = DotDict(self.dict_from_class())

            # delete unnecessary stuff
            del params.optimization_options
            del params.param_grid

            # make sure to add the name of the model
            params.MODELS = model

            # instantiate model development class
            ML = ModelDevelopment(data, params)

           # evaluate the combination of parameter values
            if self.VALIDATION_OR_TEST == "validation":
                ML.cross_validate_all_models()
            else:
                ML.tts_finalize()

            # retrieve the model and it's metric scores
            trained_model = ML.model_list.models[0]
            new_AUC = trained_model.results.mean_auc
            new_balanced_accuracy = trained_model.results.mean_balanced_accuracy

            # if the AUC is better than the current best AUC
            if new_AUC > current_best_AUC:
                print(f'Improvement! The best AUC is now {new_AUC}')
                # set current best AUC to new AUC
                current_best_AUC = new_AUC
                # store the param values
                best_params_AUC = params

            # if the balanced accuracy is better than the current best balanced accuracy:
            if new_balanced_accuracy > current_best_balanced_accuracy:
                print(f'Improvement! The best balanced accuracy is now {new_balanced_accuracy}')
                # set the current best ba to the new ba
                current_best_balanced_accuracy = new_balanced_accuracy
                # store the param values
                best_params_balanced_accuracy = params

            # if both the AUC and ba are better than the current best scores
            if new_AUC >= current_best_AUC and new_balanced_accuracy >= current_best_balanced_accuracy:
                print(f'Ultimate improvement! The best combo is now {new_AUC}, {new_balanced_accuracy} ')

                # set the current best scores to the new scores
                current_best_AUC = new_AUC
                current_best_balanced_accuracy = new_balanced_accuracy

                # store the params
                best_params_combo = params
                best_params_AUC = params
                best_params_balanced_accuracy = params

        print('========================================================')
        print(f'Finished \n Best auc : {current_best_AUC}, Best ba: {current_best_balanced_accuracy} \n Best params : {best_params_combo}')

        # set parameters in instance to best found parameters
        self.update(best_params_balanced_accuracy)

        # store the best found parameter value combinations to json
        with open(f'Results/Params/basic_params_combo_{self.DATA_TYPE}_{self.PREDICTION_TYPE}_{model[0]}_{self.TEST_OR_FINAL}.json', 'w') as fp:
            json.dump(best_params_combo, fp, cls=NpEncoder)

        with open(f'Results/Params/basic_params_AUC_{self.DATA_TYPE}_{self.PREDICTION_TYPE}_{model[0]}_{self.TEST_OR_FINAL}.json', 'w') as fp:
            json.dump(best_params_AUC, fp, cls=NpEncoder)

        with open(f'Results/Params/basic_params_balanced_accuracy_{self.DATA_TYPE}_{self.PREDICTION_TYPE}_{model[0]}_{self.TEST_OR_FINAL}.json', 'w') as fp:
            json.dump(best_params_balanced_accuracy, fp, cls=NpEncoder)

        # return the best parameter value combination and metric scores
        return best_params_combo, best_params_AUC, best_params_balanced_accuracy

    def update(self,newdata):
        for key, value in newdata.items():
            setattr(self, key, value)

    def dict_from_class(self):
        return vars(self)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

parameters = Parameters()
