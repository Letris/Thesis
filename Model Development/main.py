from data_classes import KneeData
from data_classes_hip import HipData
from data_classes_shoulder import ShoulderData
from ml_classes import ModelDevelopment
from parameters import parameters as params
import sys
import pandas as pd

if __name__ == '__main__':

    # knee diagnosis and treatment prediction modelling
    if params.DATA_TYPE == 'Knee' :

        # define data paths
        data_path = 'Data/ML data/predict_data.xlsx'
        data_old_path = 'Data/ML data/AI - Knie oud compleet.xlsx'

        # load the data
        data_new = pd.read_excel(data_path)
        data_old = pd.read_excel(data_old_path, skiprows=1)

        # diagnosis prediction
        if params.PREDICTION_TYPE == 'Diagnosis':
            data = KneeData(data_new, data_old, target_column='diagnose', drop_column='behandeling\ncategorisch',
                                to_remove_infrequent_targets=False, to_merge_infrequent_targets=True)

        # treatment prediction
        else:
            data = KneeData(data_new, data_old, target_column='behandeling\ncategorisch', drop_column='diagnose',
                                 to_remove_infrequent_targets=False, to_merge_infrequent_targets=False)

    # hip diagnosis and treatment prediction modelling
    elif params.DATA_TYPE == 'Hip':

        # define data paths
        data_path = 'Data/ML data/predict_data.xlsx'
        data_old_path = 'Data/ML data/AI - Heup oud compleet.xlsx'

        # load the data
        data_new = pd.read_excel(data_path)
        data_old = pd.read_excel(data_old_path, skiprows=1)

        # diagnosis prediction
        if params.PREDICTION_TYPE == 'Diagnosis':
            data = HipData(data_new, data_old, target_column='diagnose', drop_column='behandeling\ncategorisch',
                               to_remove_infrequent_targets=False, to_merge_infrequent_targets=True)

        # treatment prediction
        else:
            data = HipData(data_new, data_old, target_column='behandeling\ncategorisch', drop_column='diagnose',
                                   to_remove_infrequent_targets=False, to_merge_infrequent_targets=False)

    # shoulder prediction
    elif params.DATA_TYPE == 'Shoulder':
        data = pd.read_excel('Data/ML data/AI - Schouder oud compleet.xlsx')

        data = ShoulderData(data, target_column='Julie diagnose')
    else:
        print('Invalid experiment')
        sys.exit(0)

    # preprocess the data
    data.preprocess()

    # instantiate the ML class
    ML = ModelDevelopment(data, params)

    # running the experiments with one of the function below
    # ML.pipeline()
    ML.tts_finalize()
    # ML.cross_validate_all_models()
    # # ML.hp_tuning()
    # ML.finalize_and_save_models()
