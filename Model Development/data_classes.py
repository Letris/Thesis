import copy
import re
import sys
from pickle import load

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

from parameters import parameters as params

# import featuretools as ft


class Dataset:
    """Class to hold a dataset and functions to preprocess it
    """

    def __init__(self, data_new, old_data, target_column, cutoff, top_x_targets, to_merge_infrequent_targets, to_remove_infrequent_targets):
        """Initialise the instance

        Args:
            data_new (dataframe): data in the new questionnaire format
            old_data (dataframe): data in the old questionnaire format
            target_column (str): the name of the column with labels to predict
            cutoff (int): the cutoff for the allowed number of missing values in a column
            top_x_targets (int): the top x most frequently occuring targets to keep
            to_merge_infrequent_targets (bool): to merge the infrequently occuring targets to a single class or not
            to_remove_infrequent_targets (bool): to remove data with infrequently occuring targets or not
        """
        # the dataset
        self.original_data = data_new
        self.old_data = old_data
        # make a copy of the dataset
        self.learning_data = copy.deepcopy(self.original_data)
        # the cutoff for the allowed number of missing values in a column
        self.cutoff = cutoff
        # the top x most frequently occuring targets to keep
        self.top_x_targets = top_x_targets
        # to merge the infrequently occuring targets to a single class or not
        self.to_merge_infrequent_targets = to_merge_infrequent_targets
        # to remove data with infrequently occuring targets or not
        self.to_remove_infrequent_targets = to_remove_infrequent_targets
        # the name of the column with labels to predict
        self.target_column = target_column

    def remove_bad_rows(self):
        """Remove rows that have too many missing values
        """
        # get the amount of missing values per column in a dict
        missing_values = dict(self.learning_data.isnull().sum())

        # find the columns that have fewer than the cutoff amount of missing values
        subset = [col for col, amount_missing in missing_values.items()
                  if amount_missing <= self.cutoff]
        # remove rows where values are missing for these columns
        self.learning_data = self.learning_data.dropna(
            how='any', subset=subset)
        self.print_length_data('4 After dropping rows with missing values')
        # remove rows with impossible age
        self.learning_data = self.learning_data[(
            self.learning_data['leeftijd'] < 110) & (self.learning_data['leeftijd'] > 18)]
        self.print_length_data('5 After removing invalid age')
        # remove rows with impossible length
        self.learning_data = self.learning_data[(
            self.learning_data['lengte'] < 220) & (self.learning_data['lengte'] > 100)]
        self.print_length_data('6 After removing invalid length')
        # remove rows with impossible length
        self.learning_data = self.learning_data[(
            self.learning_data['gewicht'] < 200) & (self.learning_data['gewicht'] > 50)]
        self.print_length_data('7 after removing invalid weight')

    def print_length_data(self, message):
        """Print the length of the data

        Args:
            message (str): the message to add
        """
        print(f'{message} {len(self.learning_data)}')

    def merge_infrequent_targets(self):
        """Merge all targets except the top x most occurring targets
        """
        # count the targets
        value_counts = self.learning_data[self.target_column].value_counts(
        ).to_frame()
        # find the top x most occuring targets
        top_targets = value_counts.head(self.top_x_targets).index.to_list()
        # change targets that appear infrequently to 'other'
        self.learning_data[self.target_column] = self.learning_data[self.target_column].apply(
            lambda x: self.combine_targets(x, top_targets))

    def remove_infrequent_targets(self, cutoff=50):
        """Remove rows from the dataset with a target that appears than the cutoff in the entire dataset

        Args:
            cutoff (int, optional): The cutoff value. Defaults to 50.
        """
        # count the targets
        value_counts = self.learning_data[self.target_column].value_counts(
        ).to_dict()

        # remove rows with targets that appear less often than the cutoff
        for target, count in value_counts.items():
            if count <= cutoff:
                self.learning_data = self.learning_data[self.learning_data[self.target_column] != target]

    def combine_targets(self, target, top_targets):
        """Combine infrequently occuring targets into a single class

        Args:
            target (str): the target to check
            top_targets (list): the top n most occuring targets

        Returns:
            str: original target value or combined value
        """

        if target in top_targets:
            return target
        else:
            return 'Anders'

    def process_nominal(self):
        """Preprocess columns with nominal values
        """

        # find the missing values
        missing_values = dict(
            self.learning_data[self.nominal_cols].isnull().sum())

        # replace column values with 0 if the amount of missing values is higher than the cutoff
        for col, amount_missing in missing_values.items():
            if amount_missing > self.cutoff:
                self.learning_data[col] = self.learning_data[col].replace(
                    np.nan, 0)

        # change type to string (nominal)
        self.learning_data[self.nominal_cols] = self.learning_data[self.nominal_cols].astype(
            str)

        # one-hot-encode the nominal columns
        self.learning_data = pd.get_dummies(
            self.learning_data, columns=self.nominal_cols)

    def process_ordinal(self):
        """Preprocess columns with ordinal values
        """
        # replace missing values with 0
        self.learning_data[self.ordinal_cols] = self.learning_data[self.ordinal_cols].replace(
            np.nan, 0)

    def encode_labels(self):
        """Encode labels numerically
        """
        self.le = preprocessing.LabelEncoder()
        transformed_labels = self.le.fit_transform(
            self.learning_data[self.target_column])
        self.learning_data[self.target_column] = transformed_labels

    def split_targets_to_separate_datasets(self):
        """Function to create separate datasets for each target

        Returns:
            dict: {target : dataset}
        """
        # change some column types to prevent errors
        self.change_column_types()
        # remove rows with invalid data
        self.remove_invalid_target_rows()

        if self.to_merge_infrequent_targets:
            # merge targets that are infrequent to other
            self.merge_infrequent_targets()
        if self.to_remove_infrequent_targets:
            self.remove_infrequent_targets()

        datasets = {}

        for target, group in self.learning_data.groupby(self.target_column):
            datasets[target] = group

        return datasets

    def missing_values_summary(self):
        """Generate missing values summary and store it to excel
        """
        # Number of missing in each column
        missing = pd.DataFrame(self.learning_data.isnull().sum()).rename(
            columns={0: 'total'})

        # Create a percentage missing
        missing['percent'] = missing['total'] / len(self.learning_data)

        missing.to_excel('Data/missing_values_knee.xlsx')

    @staticmethod
    def load_excel(path):
        return pd.read_excel(path)

    def prepare_data(self):
        """Combine the old and new data sets
        """

        # preprocess the new data set to match with the old data set
        self.preprocess_new()

        # preprocess the old data set to match with the new data set
        self.preprocess_old()

        # check if both data sets now contain the same columns
        self.check_equal_columns()

        # combine the old and new data sets
        self.learning_data = pd.concat(
            [self.old_data, self.learning_data], axis=0)

    def check_equal_columns(self):
        """Check if the old and new data sets have the same columns
        """
        # get the set of columns in the new data set
        set_a = set(self.learning_data.columns.to_list())

        # get the set of columns in the old data set
        set_b = set(self.old_data.columns.to_list())

        # get the intersection of the two set
        intersect = set_a.intersection(set_b)

        # check if all lengths are equal
        print(len(set_a), len(set_b), len(intersect))

        # print the mismatching columns if there are any
        print(set_a.symmetric_difference(set_b))


class KneeData(Dataset):

    def __init__(self, data_new, data_old, target_column='diagnose', drop_column='behandeling\ncategorisch', cutoff=10, top_x_targets=2,
                 to_merge_infrequent_targets=False, to_remove_infrequent_targets=True):

        super().__init__(data_new, data_old, target_column, cutoff, top_x_targets,
                         to_merge_infrequent_targets, to_remove_infrequent_targets)

        # columns with nominal data
        self.nominal_cols = ['zijde_klachten', 'hydrops_nu', 'locatie_pijn', 'type_knie_trauma', 'instabiliteit', 'patellaluxatie', 'staken_werk_hobby_sport',
                             'afwijkend_looppatroon', 'liespijn', 'fysio_en_effect', 'werking_pijnstillers', 'afstand_mobiliseren', 'flexie', 'extensie', 'startstijfheid',
                             'infiltratie', 'wens_operatie', 'type_operatie', 'belastbaarheid_na_trauma', 'behandelaar', 'progressie', 'pijn_vs_bewegingsbeperking']

        # columns with ordinal data
        self.ordinal_cols = ['K&L', 'duur_klachten', 'trauma_timing',
                             'NRS_pijn_in_rust', 'NRS_bij_bewegen', 'wens_operatie']

        # columns with binary data
        self.binary_cols = ['VGOK_ipsilateraal_type_1', 'VGOK_ipsilateraal_type_2', 'VGOK_ipsilateraal_type_3',
                            'VGOK_ipsilateraal_type_4', 'VGOK_ipsilateraal_type_5', 'VGOK_ipsilateraal_type_6',
                            'VGOK_ipsilateraal_type_7', 'VGOK_ipsilateraal_type_8', 'VGOK_ipsilateraal_type_9', 'VGOK_contralateraal', 'trauma',
                            'bekende_rugklachten', 'rug_operatie', 'werk_ja_nee',
                            'OAS', 'antibiotica', 'roken', 'VGOK_contralateraal_type_1', 'VGOK_contralateraal_type_2',
                            'VGOK_contralateraal_type_3', 'VGOK_contralateraal_type_4', 'VGOK_contralateraal_type_5',
                            'VGOK_contralateraal_type_6', 'VGOK_contralateraal_type_7', 'VGOK_contralateraal_type_8',
                            'VGOK_contralateraal_type_9', 'type_pijnstilling_1', 'type_pijnstilling_2',
                            'type_pijnstilling_3', 'type_pijnstilling_4', 'type_pijnstilling_5']

        # columns with continuous data
        self.cont_cols = ['gewicht', 'leeftijd', 'lengte',
                          'NRS_pijn_in_rust', 'NRS_bij_bewegen']

        # columns with text
        self.text_cols = ['Hulpvraag', 'type_werk',
                          'welke_hobby_sport', 'overig']

        # the target columns
        self.target_column = target_column

        # the target column to drop
        self.drop_column = drop_column

        # type of data
        self.DATA_NAME = 'knee'

    def preprocess(self):
        """Preprocess the learning data
        """

        # combine the old and new data sets
        self.prepare_data()

        # create a summary of the missing values
        self.missing_values_summary()

        # change some column types to prevent errors
        self.change_column_types()

        # remove rows with invalid data
        self.remove_invalid_target_rows()

        # remove rows with missing values
        self.remove_bad_rows()

        # replace the remaining missing values with 0 (missing by design)
        self.learning_data.fillna(0, inplace=True)

        # fix some anomalies in the data
        self.fix_anomolies()

        # merge or remove infrequent targets
        if self.to_merge_infrequent_targets:
            # merge targets that are infrequent to other
            self.merge_infrequent_targets()
        if self.to_remove_infrequent_targets:
            self.remove_infrequent_targets()

        # preprocess different data types
        self.process_ordinal()
        self.process_nominal()
        self.process_binary()

        # remove some NA's from the target column (only for treatments)
        if self.target_column != 'diagnosis':
            self.remove_nan_from_target()

        # numerically encode the labels
        self.encode_labels()

        # split the data in data and labels
        self.X, self.y = self.split_to_XY(self.learning_data)

        # check the data for normality
        self.norm_test()

        # store the data and labels to excel
        self.X.to_excel('X_knee_treatment.xlsx')
        self.y.to_excel('y_knee.xlsx')

    def norm_test(self):
        """Test each column for normality
        """
        for col in self.X.columns.to_list():
            to_test = self.X[col]
            is_normal = stats.shapiro(to_test)
            print(f'{to_test} {is_normal}')

    def remove_nan_from_target(self):
        """Remove rows with unknown target
        """
        self.learning_data = self.learning_data.loc[self.learning_data[self.target_column] != 'nan']

    def fix_anomolies(self):
        """Fix some anomalies in the data
        """
        self.learning_data['wens_operatie'] = self.learning_data['wens_operatie'].map(
            {'niet': 1, 'misschien': 2, 'wel': 3})
        self.learning_data['locatie_pijn'] = self.learning_data['locatie_pijn'].map({'Aan de buitenkant van de knie': 1,
                                                                                     'Aan de binnenkant van de knie': 2, 'Bovenop de knieschijf': 3, 'Over de kniepees': 4, 'Achter de knieschijf': 5})

    def preprocess_new(self):
        """Preprocess the data that was collected with the updates questionnaire to match with the data collected with the old questionnaire
        """

        # find all columns in the data that relate to knee complaints
        knee__prom_cols = [
            col for col in self.learning_data.columns if 't0_ORT_K' in col]

        # define the columns that contain the targets
        knee_outcome_cols = ['Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__K_KL__vraagsscore',
                             'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__K_diagnose__vraagsscore',
                             'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__K_behandeling__vraagsscore']

        # define the remaining demographic columns to include
        keep_in_both = ['patientnummer', 'zorgverleners', 't0_ORT_leeftijd_antwoord_waarde', 't0_ORT_lengte_antwoord_waarde',
                        't0_ORT_gewicht_antwoord_waarde', 't0_ORT_werk_antwoord_optie', 't0_ORT_type_werk_antwoord_tekst',
                        't0_ORT_welke_hobby_sport_antwoord_tekst', 't0_ORT_staken_werk_hobby_sport_knie_antwoord_optie',
                        't0_ORT_OAS_antwoord_optie', 't0_ORT_type_OAS_antwoord_tekst', 't0_ORT_antibiotica_antwoord_optie',
                        't0_ORT_roken_antwoord_optie']

        # define the complete list of columns to keep
        knee_cols = keep_in_both + knee__prom_cols + knee_outcome_cols

        # keep only data that was labeled by the orthopedic surgeons
        knee_data = self.learning_data[self.learning_data['Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__type_gewricht__2_antwoord_optie'] == 'Ja']

        # keep only the columns that are relevant
        knee_data = knee_data[knee_cols]

        # rename the target columns
        knee_data.rename({'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__K_KL__vraagsscore': 'K&L',
                          'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__K_diagnose__vraagsscore': 'diagnose',
                          'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__K_behandeling__vraagsscore': 'behandeling'}, inplace=True, axis=1)

        # map the numeric representation of the diagnoses to their textual representation
        diagnoses = {1: 'Gonartrose', 2: 'Patellofemorale artrose', 3: 'Meniscusletsel', 4: 'Patellofemoraal pijnsyndroom',
                     5: 'Enthesopathie', 6: 'Bakerse cyste', 7: 'Voorste kruisbandlaesie', 8: 'Mediale collateraalbandlaesie',
                     9: 'Laterale collateraalbandlaesie', 10: 'Achterse kruisbandlaesie', 11: 'Posttraumatische afwijking (geen artrose)',
                     12: 'Bursitis: rondom de knie', 13: 'Patella instabiliteit', 14: 'Tumor: benigne en maligne', 15: 'Osgood-Schlatter',
                     16: 'Corpus liberum', 17: 'Osteochondraal defect ', 18: '(Rheumatoide) Arthritis', 19: 'Kristalarthropathy',
                     20: 'Cyclops (geisoleerd)', 21: 'TLWK | SI gerelateerd', 22: 'Distorsie | Contusie', 23: 'Osteochondritis dissecans',
                     24: 'Peesruptuur: quadricepspees | patellapees', 25: 'Standsafwijking', 26: 'Klachten na TKP', 27: 'Geen diagnose'}

        knee_data['diagnose'] = knee_data['diagnose'].map(diagnoses)

        # map the numeric representation of the treatments to their textual represenation
        behandelingen = {1: 'Expectatief', 2: 'Fysiotherapie | Leefstijladviezen', 3: 'NSAIDs', 4: 'Intra-articulaire infiltratie',
                         5: 'Extra-articulaire infiltratie', 6: 'Hulpmiddelen: steunzolen | brace', 7: 'Nucleair onderzoek: botscan | SPECT/CT',
                         8: 'MRI', 9: 'Echo', 10: 'CT', 11: 'Knie prothese: UKP | TKP', 12: 'Arthroscopie: meniscectomie | meniscopexie | ligamentaire reconstructie',
                         13: 'Revisie TKP', 14: 'Tibiakoposteotomie | HTO', 15: 'Arthrotomie: bijv repair osteochondraal defect',
                         16: 'MPFL reconstructie', 17: 'Pees reconstructie: quadricepspees repair | patellapees repair',
                         18: 'Resectie: Bursa | Osgood-Schlatter | exostose', 19: 'Verwijderen osteosynthese materiaal',
                         20: 'Verwijzing ander specialisme | Verwijzing voor second opinion', 21: 'Overig',
                         22: 'Niet van toepassing', 23: 'Gegevens opvragen'}

        knee_data['behandeling'] = knee_data['behandeling'].map(behandelingen)

        # assign specific treatments to the surgery or conservative class
        knee_data['behandeling\ncategorisch'] = knee_data['behandeling'].apply(
            self.assign_cat)

        # remove a whole lot of excess text from the columns
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('t0_ORT_K_', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('t0_ORT_', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('_antwoord_waarde', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('_antwoord_tekst', '', x))
        knee_data = knee_data.rename(columns=lambda x: re.sub('__', '_', x))

        # drop all score columns
        knee_data = knee_data[knee_data.columns.drop(
            list(knee_data.filter(regex='vraagsscore')))]

        # rename a few more columns to match with the columns in the old data set
        knee_data.rename({'zorgverleners': 'behandelaar',
                         'patientnummer': 'Nummer'}, axis=1, inplace=True)
        knee_data.rename(
            {'werk': 'werk_ja_nee', 'staken_werk_hobby_sport_knie': 'staken_werk_hobby_sport'}, axis=1, inplace=True)

        # drop some columns that are not present in the old data set
        knee_data.drop(['nachtpijn', 'gestrekt_heffen', 'jicht_arthritis_2',
                       'jicht_arthritis_3', 'jicht_arthritis_1'], axis=1, inplace=True)

    @staticmethod
    def assign_cat(treatment):
        """Assign a treatment to either the surgical or conservative treatment category

        Args:
            treatment (str): the treatment to assign

        Returns:
            str: the category of the treatment
        """

        # define operative treatments
        operatieve_behandelingen = ['Knie prothese: UKP | TKP', 'Arthroscopie: meniscectomie | meniscopexie | ligamentaire reconstructie',
                                    'Revisie TKP', 'Tibiakoposteotomie | HTO', 'Arthrotomie: bijv repair osteochondraal defect',
                                    'MPFL reconstructie', 'Pees reconstructie: quadricepspees repair | patellapees repair',
                                    'Resectie: Bursa | Osgood-Schlatter | exostose', 'Verwijderen osteosynthese materiaal']

        # if treatment is an operative treatment
        if treatment in operatieve_behandelingen:
            # assign to operative class
            return 'Operatief'
        else:
            # else assign to conservative class
            return 'Conservatief'

    def preprocess_old(self):
        """Preprocess the data that was collected with the old questionnaire to match with the data collected with the updated questionnaire
        """

        # remove excess text from columns
        knee_old = self.old_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('ORT__K_', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('__', '_', x))

        # drop some columns that are not in the new data set
        knee_old.drop(['krachtsverlies', 'paresthesie',
                      'sensibiliteit', 'type_antibiotica'], axis=1, inplace=True)

        # assign treatments to either the surgical or conservative class
        knee_old['behandeling\ncategorisch'] = knee_old['behandeling\ncategorisch'].map({'Aanvullend onderzoek': 'Conservatief', 'NVT': 'Conservatief',
                                                                                         'Verwijzing': 'Conservatief', 'Conservatief': 'Conservatief', 'Operatief': 'Operatief'})

        self.old_data = knee_old

    def change_column_types(self):
        """Change the types of some columns to prevent errors
        """
        # Changing column types
        self.learning_data[self.target_column] = self.learning_data[self.target_column].astype(
            str)
        self.learning_data['K&L'] = pd.to_numeric(
            self.learning_data['K&L'], errors='coerce')
        self.learning_data['antibiotica'] = pd.to_numeric(
            self.learning_data['antibiotica'], errors='coerce')
        self.learning_data['overig'] = self.learning_data['overig'].astype(str)

    def remove_invalid_target_rows(self):
        """Remove rows in learning data where the diagnosis is missing or contains an invalid character
        """
        self.print_length_data('1. start')
        # remove nans
        self.data = self.learning_data[self.learning_data[self.target_column] != 'nan']
        self.print_length_data('2. After removing nan from targets')
        # remove rows with numeric characters
        self.data = self.learning_data[~self.learning_data[self.target_column].str.contains(
            r'[0-9]')]
        self.print_length_data('3. After removing invalid targets')

    def process_binary(self):
        """Preprocess the binary columns
        """
        # coerce binary columns to string
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].astype(
            str)
        # map all binary values to 0 or 1
        map_dict = {'Ja': 1, 'Nee': 0, '1.0': 0, '2.0': 1, '1': 0, '2': 1}
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].applymap(
            map_dict.get)

    def split_to_XY(self, data):
        """Split the data into data and labels

        Args:
            data (pandas dataframe): the preprocessed data

        Returns:
            X, y: data, labels
        """

        # drop a few duplicate rows
        data.drop_duplicates(subset='Nummer', inplace=True)

        # drop columns that won't be used and the target column to get the final data set
        X = data.drop([self.target_column, self.drop_column, 'Nummer', 'behandeling', 'Hulpvraag', 'K&L',
                       'type_werk', 'welke_hobby_sport',
                       'type_OAS', 'overig', 'OAS'], axis=1)

        # get the labels of the data
        y = data[self.target_column]

        return X, y


class KneeDataPredict(Dataset):
    """Class to hold the and preprocess the data for prediction in a production environment
    """

    def __init__(self, data_new):
        """Initialise the instance

        Args:
            data_new (dataframe): the new data containing the patient to predict
        """
        self.new_data = data_new
        # still need to load the old data for preprocessing to work
        self.old_data = pd.read_excel(
            'Data/ML data/AI - Knie oud compleet.xlsx', skiprows=1)

        self.DATA_NAME = 'knee'
        self.cutoff = 10

        # columns with nominal data
        self.nominal_cols = ['zijde_klachten', 'hydrops_nu', 'locatie_pijn', 'type_knie_trauma', 'instabiliteit', 'patellaluxatie', 'staken_werk_hobby_sport',
                             'afwijkend_looppatroon', 'liespijn', 'fysio_en_effect', 'werking_pijnstillers', 'afstand_mobiliseren', 'flexie', 'extensie', 'startstijfheid',
                             'infiltratie', 'wens_operatie', 'type_operatie', 'belastbaarheid_na_trauma', 'behandelaar', 'progressie', 'pijn_vs_bewegingsbeperking']

        # columns with ordinal data
        self.ordinal_cols = ['K&L', 'duur_klachten', 'trauma_timing',
                             'NRS_pijn_in_rust', 'NRS_bij_bewegen', 'wens_operatie']

        # columns with binary data
        self.binary_cols = ['VGOK_ipsilateraal_type_1', 'VGOK_ipsilateraal_type_2', 'VGOK_ipsilateraal_type_3',
                            'VGOK_ipsilateraal_type_4', 'VGOK_ipsilateraal_type_5', 'VGOK_ipsilateraal_type_6',
                            'VGOK_ipsilateraal_type_7', 'VGOK_ipsilateraal_type_8', 'VGOK_ipsilateraal_type_9', 'VGOK_contralateraal', 'trauma',
                            'bekende_rugklachten', 'rug_operatie', 'werk_ja_nee',
                            'OAS', 'antibiotica', 'roken', 'VGOK_contralateraal_type_1', 'VGOK_contralateraal_type_2',
                            'VGOK_contralateraal_type_3', 'VGOK_contralateraal_type_4', 'VGOK_contralateraal_type_5',
                            'VGOK_contralateraal_type_6', 'VGOK_contralateraal_type_7', 'VGOK_contralateraal_type_8',
                            'VGOK_contralateraal_type_9', 'type_pijnstilling_1', 'type_pijnstilling_2',
                            'type_pijnstilling_3', 'type_pijnstilling_4', 'type_pijnstilling_5']

        # columns with continuous data
        self.cont_cols = ['gewicht', 'leeftijd', 'lengte',
                          'NRS_pijn_in_rust', 'NRS_bij_bewegen']

        # columns with text
        self.text_cols = ['Hulpvraag', 'type_werk',
                          'welke_hobby_sport', 'overig']

    def preprocess(self):
        """Preprocess the learning data
        """

        # combine the old and new data sets
        self.prepare_data()

        # remove rows with missing values
        self.remove_bad_rows()

        self.learning_data.fillna(0, inplace=True)

        self.fix_anomolies()

        # preprocess different data types
        self.process_ordinal()
        self.process_nominal()
        self.process_binary()

        # split the data in data and labels
        self.X, self.y = self.split_to_XY(self.learning_data)

    def fix_anomolies(self):
        self.learning_data['wens_operatie'] = self.learning_data['wens_operatie'].map(
            {'niet': 1, 'misschien': 2, 'wel': 3})
        self.learning_data['locatie_pijn'] = self.learning_data['locatie_pijn'].map({'Aan de buitenkant van de knie': 1,
                                                                                     'Aan de binnenkant van de knie': 2, 'Bovenop de knieschijf': 3, 'Over de kniepees': 4, 'Achter de knieschijf': 5})

    def prepare_data(self):
        self.preprocess_new()
        self.preprocess_old()
        self.check_equal_columns()
        self.learning_data = pd.concat(
            [self.old_data, self.learning_data], axis=0)

    def preprocess_new(self):

        cols = [col for col in list(
            self.learning_data.columns) if not isinstance(col, int)]
        knee__prom_cols = [col for col in cols if 't0_ORT_K' in col]

        keep_in_both = ['patientnummer', 'zorgverleners', 't0_ORT_leeftijd_antwoord_waarde', 't0_ORT_lengte_antwoord_waarde',
                        't0_ORT_gewicht_antwoord_waarde', 't0_ORT_werk_antwoord_optie', 't0_ORT_type_werk_antwoord_tekst',
                        't0_ORT_welke_hobby_sport_antwoord_tekst', 't0_ORT_staken_werk_hobby_sport_knie_antwoord_optie',
                        't0_ORT_OAS_antwoord_optie', 't0_ORT_type_OAS_antwoord_tekst', 't0_ORT_antibiotica_antwoord_optie',
                        't0_ORT_roken_antwoord_optie']

        knee_cols = keep_in_both + knee__prom_cols

        knee_data = self.learning_data[self.learning_data['t0_ORT_locatie_klachten_5_antwoord_optie'] == 'Ja']

        knee_data = knee_data[knee_cols]

        knee_data = knee_data.rename(
            columns=lambda x: re.sub('t0_ORT_K_', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('t0_ORT_', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('_antwoord_waarde', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        knee_data = knee_data.rename(
            columns=lambda x: re.sub('_antwoord_tekst', '', x))
        knee_data = knee_data.rename(columns=lambda x: re.sub('__', '_', x))
        knee_data = knee_data[knee_data.columns.drop(
            list(knee_data.filter(regex='vraagsscore')))]
        knee_data.rename({'zorgverleners': 'behandelaar',
                         'patientnummer': 'Nummer'}, axis=1, inplace=True)
        knee_data.rename(
            {'werk': 'werk_ja_nee', 'staken_werk_hobby_sport_knie': 'staken_werk_hobby_sport'}, axis=1, inplace=True)
        knee_data.drop(['nachtpijn', 'gestrekt_heffen', 'jicht_arthritis_2',
                       'jicht_arthritis_3', 'jicht_arthritis_1'], axis=1, inplace=True)
        self.learning_data = knee_data

    def preprocess_old(self):
        knee_old = self.old_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('ORT__K_', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('__', '_', x))
        knee_old.drop(['krachtsverlies', 'paresthesie', 'sensibiliteit', 'type_antibiotica', 'diagnose',
                       'K&L', 'behandeling', 'behandeling\ncategorisch'], axis=1, inplace=True)
        self.old_data = knee_old

    def check_equal_columns(self):
        set_a = set(self.learning_data.columns.to_list())
        set_b = set(self.old_data.columns.to_list())
        intersect = set_a.intersection(set_b)

        print(len(set_a), len(set_b), len(intersect))
        print(set_a.symmetric_difference(set_b))

    def process_continuous(self):
        for col in self.cont_cols:
            labels = [f'{col}_bin_{i}' for i in range(10)]
            self.learning_data[f'{col}_binned'], bins = pd.cut(
                self.learning_data[col], bins=10, retbins=True, labels=labels, duplicates='drop')
            new_labels = [f'{col}_{bin}' for bin in bins]
            label_map = dict(zip(labels, new_labels))
            self.learning_data[f'{col}_binned'] = self.learning_data[f'{col}_binned'].map(
                label_map)
            self.learning_data = pd.get_dummies(
                self.learning_data, columns=[f'{col}_binned'])
            # self.learning_data.drop([col], axis=1, inplace=True)

    def change_column_types(self):
        """Change the types of some columns to prevent errors
        """
        # Changing column types
        self.learning_data[self.target_column] = self.learning_data[self.target_column].astype(
            str)
        self.learning_data['K&L'] = pd.to_numeric(
            self.learning_data['K&L'], errors='coerce')
        self.learning_data['antibiotica'] = pd.to_numeric(
            self.learning_data['antibiotica'], errors='coerce')
        self.learning_data['overig'] = self.learning_data['overig'].astype(str)

    def remove_invalid_target_rows(self):
        """Remove rows in learning data where the diagnosis is missing or contains an invalid character
        """
        # remove nans
        self.data = self.learning_data[self.learning_data[self.target_column] != 'nan']

        # remove rows with numeric characters
        self.data = self.learning_data[~self.learning_data[self.target_column].str.contains(
            r'[0-9]')]

    def process_binary(self):
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].astype(
            str)
        map_dict = {'Ja': 1, 'Nee': 0, '1.0': 0, '2.0': 1, '1': 0, '2': 1}
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].applymap(
            map_dict.get)

    def split_to_XY(self, data):
        data.drop_duplicates(subset='Nummer', inplace=True, keep='last')
        behandelaars = ['behandelaar_Mw. E. Bos', 'behandelaar_0', 'behandelaar_dr. Bogie (Rob), dr. van Zoest (Wart)',
                        'behandelaar_dr. van Zoest (Wart), Dr. R. Agricola (Rintje)']
        X = data.drop(['Hulpvraag', 'jicht_arthritis_4',
                       'type_werk', 'welke_hobby_sport',
                       'type_OAS', 'overig', 'OAS'] + behandelaars, axis=1)

        y = None

        return X, y


class HipData(Dataset):

    def __init__(self, data_new, data_old, target_column='diagnose', drop_column='behandeling\ncategorisch', cutoff=10, top_x_targets=2,
                 to_merge_infrequent_targets=True, to_remove_infrequent_targets=False):

        super().__init__(data_new, data_old, target_column, cutoff, top_x_targets,
                         to_merge_infrequent_targets, to_remove_infrequent_targets)

        self.nominal_cols = ['behandelaar', 'staken_werk_hobby_sport_heup', 'zijde', 'locatie_pijn', 'pijn_vs_bewegingsbeperking', 'sokken_schoenen', 'traplopen',
                             'looppatroon', 'knieklachten', 'fysiotherapie_en_effect', 'type_behandeling', 'wens_operatie', 'progressie', 'afstand_mobiliseren', 'stijfheid']

        self.ordinal_cols = ['duur_klacht']

        self.binary_cols = ['werk_ja_nee', 'antibiotica', 'roken', 'VGOK_ipsilateraal', 'rugklachten', 'VGOK_contralateraal',
                            'type_pijnstilling_1', 'type_pijnstilling_2', 'type_pijnstilling_3', 'type_pijnstilling_4', 'type_pijnstilling_5', 'effect_pijnstilling']

        self.cont_cols = ['gewicht', 'leeftijd', 'lengte', 'NRS_in_rust', 'NRS_bij_bewegen',
                          ]

        self.text_cols = ['Hulpvraag', 'type_werk',
                          'welke_hobby_sport', 'overig_opmerkingen']

        self.target_column = target_column
        self.drop_column = drop_column
        self.DATA_NAME = 'knee'

    def preprocess(self):
        """Preprocess the learning data
        """

        self.prepare_data()

        self.missing_values_summary()

        # change some column types to prevent errors
        self.change_column_types()
        # remove rows with invalid data
        self.remove_invalid_target_rows()
        # remove rows with missing values
        self.remove_bad_rows()

        self.learning_data.fillna(0, inplace=True)

        if self.to_merge_infrequent_targets:
            # merge targets that are infrequent to other
            self.merge_infrequent_targets()
        if self.to_remove_infrequent_targets:
            self.remove_infrequent_targets()

        # preprocess different data types
        self.process_ordinal()
        self.process_nominal()
        self.process_binary()

        # encode the labels to float
        self.encode_labels()

        # split the data in data and labels
        self.X, self.y = self.split_to_XY(self.learning_data)

    def preprocess_new(self):
        hip__prom_cols = [
            col for col in self.learning_data.columns if 't0_ORT_H' in col]

        hip_outcome_cols = ['Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__H_KL__vraagsscore',
                            'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__H_diagnose__vraagsscore',
                            'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__H_behandeling__vraagsscore']

        keep_in_both = ['patientnummer', 'zorgverleners', 't0_ORT_leeftijd_antwoord_waarde', 't0_ORT_lengte_antwoord_waarde',
                        't0_ORT_gewicht_antwoord_waarde', 't0_ORT_werk_antwoord_optie', 't0_ORT_type_werk_antwoord_tekst',
                        't0_ORT_welke_hobby_sport_antwoord_tekst', 't0_ORT_staken_werk_hobby_sport_heup_antwoord_optie',
                        't0_ORT_OAS_antwoord_optie', 't0_ORT_type_OAS_antwoord_tekst', 't0_ORT_antibiotica_antwoord_optie',
                        't0_ORT_roken_antwoord_optie']

        hip_cols = keep_in_both + hip__prom_cols + hip_outcome_cols

        hip_data = self.learning_data[self.learning_data['Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__type_gewricht__1_antwoord_optie'] == 'Ja']

        hip_data = hip_data[hip_cols]

        hip_data.rename({'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__H_KL__vraagsscore': 'K&L',
                         'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__H_diagnose__vraagsscore': 'diagnose',
                         'Uitkomst_1e_consult_-_Heup_|_Knie_uitkomst_arts__H_behandeling__vraagsscore': 'behandeling'}, inplace=True, axis=1)

        diagnoses = {1: 'Coxartrose', 2: 'Bursitis trochanterica', 3: 'TLWK | SI', 4: 'Enthesopathie',
                     5: 'Klachten na THP', 6: 'Avasculaire necrose', 7: 'Gonartrose', 8: 'Posstraumatische afwijking',
                     9: 'Perthes', 10: 'Labrumlaesie', 11: 'Femoroacetabulaire impingement (FAI)',
                     12: '(Rheumatoide) Arthritis', 13: 'Heupdysplasie', 14: 'Stress fractuur', 15: 'Tumor: benigne en maligne',
                     16: 'Artrose os pubis', 17: 'Beenlengteverschil', 18: 'Myalgeen gerelateerde klacht', 19: 'Overig | Niet van toepassing'}

        hip_data['diagnose'] = hip_data['diagnose'].map(diagnoses)

        behandelingen = {1: 'Expectatief', 2: 'Fysiotherapie | Leefstijladviezen', 3: 'NSAIDs', 4: 'Extra-articulaire infiltratie',
                         5: 'Nucleair onderzoek: botscan | SPECT/CT', 6: 'MRI', 7: 'Echo',
                         8: 'Marcainisatie heup', 9: 'Totale heupprothese', 10: 'Revisie THP', 11: 'Verwijzing ander specialisme',
                         12: 'Intra-articulaire infiltratie van de knie',
                         13: 'Fenestratie van fascie', 14: 'Overig', 15: 'Niet van toepassing',
                         16: 'Gegevens opvragen'}

        hip_data['behandeling'] = hip_data['behandeling'].map(behandelingen)
        hip_data['behandeling\ncategorisch'] = hip_data['behandeling'].apply(
            self.assign_cat)

        hip_data = hip_data.rename(
            columns=lambda x: re.sub('t0_ORT_H_', '', x))
        hip_data = hip_data.rename(columns=lambda x: re.sub('t0_ORT_', '', x))
        hip_data = hip_data.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        hip_data = hip_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        # hip_data = hip_data.rename(columns=lambda x: re.sub('_vraagsscore','',x))
        # hip_data = hip_data.rename(columns=lambda x: re.sub('__vraagsscore','',x))
        hip_data = hip_data.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        hip_data = hip_data.rename(
            columns=lambda x: re.sub('_antwoord_waarde', '', x))
        hip_data = hip_data.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        hip_data = hip_data.rename(
            columns=lambda x: re.sub('_antwoord_tekst', '', x))
        hip_data = hip_data.rename(columns=lambda x: re.sub('__', '_', x))
        hip_data = hip_data.rename(columns=lambda x: x.strip('_'))
        # hip_data = hip_data[hip_data.columns.drop(list(hip_data.filter(regex='antwoord_optie')))]
        hip_data = hip_data[hip_data.columns.drop(
            list(hip_data.filter(regex='vraagsscore')))]
        hip_data.rename({'zorgverleners': 'behandelaar',
                        'patientnummer': 'Nummer'}, axis=1, inplace=True)
        hip_data.rename(
            {'werk': 'werk_ja_nee', 'staken_werk_hobby_sport_knie': 'staken_werk_hobby_sport'}, axis=1, inplace=True)
        hip_data.drop(['VGOK_welke_ipsi', 'VGOK_welke_contra',
                      'nachtpijn', 'rug_OK'], axis=1, inplace=True)
        hip_data['wens_operatie'] = hip_data['wens_operatie'].map(
            {'misschien': 2, 'niet': 1, 'wel': 3})
        hip_data['locatie_pijn'] = hip_data['locatie_pijn'].map({'Aan de voorkant van het bovenbeen': 1, 'In de lies': 2,
                                                                 'Over de bil/achterkant van het bovenbeen': 3, 'Aan de zijkant van het bovenbeen': 4})
        self.learning_data = hip_data

    @staticmethod
    def assign_cat(treatment):
        operatieve_behandelingen = ['Totale heupprothese', 'Revisie THP']

        if treatment in operatieve_behandelingen:
            return 'Operatief'
        else:
            return 'Conservatief'

    def preprocess_old(self):
        knee_old = self.old_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        knee_old = knee_old.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('ORT__H_', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('ORT_H_', '', x))
        knee_old = knee_old.rename(columns=lambda x: re.sub('__', '_', x))
        knee_old.drop(['krachtsverlies', 'paresthesie', 'sensibiliteit',
                      'Unnamed: 48', 'Unnamed: 49', 'type_antibiotica'], axis=1, inplace=True)
        knee_old['behandeling\ncategorisch'] = knee_old['behandeling\ncategorisch'].map({'Aanvullend onderzoek': 'Conservatief', 'NVT': 'Conservatief',
                                                                                         'Verwijzing': 'Conservatief', 'Conservatief': 'Conservatief', 'Operatief': 'Operatief', np.nan: 'Conservatief'})
        self.old_data = knee_old

    def change_column_types(self):
        """Change the types of some columns to prevent errors
        """
        # Changing column types
        self.learning_data[self.target_column] = self.learning_data[self.target_column].astype(
            str)
        self.learning_data['K&L'] = pd.to_numeric(
            self.learning_data['K&L'], errors='coerce')
        self.learning_data['antibiotica'] = pd.to_numeric(
            self.learning_data['antibiotica'], errors='coerce')

    def remove_invalid_target_rows(self):
        """Remove rows in learning data where the diagnosis is missing or contains an invalid character
        """
        self.print_length_data('1. start')
        # remove nans
        self.data = self.learning_data[self.learning_data[self.target_column] != 'nan']
        self.print_length_data('2. After removing nan from targets')
        # remove rows with numeric characters
        self.data = self.learning_data[~self.learning_data[self.target_column].str.contains(
            r'[0-9]')]
        self.print_length_data('3. After removing invalid targets')

    def process_binary(self):
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].astype(
            str)
        map_dict = {'Ja': 1, 'Nee': 0, '1.0': 0, '2.0': 1, '1': 0, '2': 1}
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].applymap(
            map_dict.get)

    def split_to_XY(self, data):
        data.drop_duplicates(subset='Nummer', inplace=True)

        X = data.drop([self.target_column, self.drop_column, 'Nummer', 'behandeling', 'Hulpvraag', 'K&L',
                       'type_werk', 'welke_hobby_sport', 'effect_pijnstilling',
                       'type_OAS', 'overig_opmerkingen', 'OAS'], axis=1)

        y = data[self.target_column]

        return X, y


class HipDataExperimental(Dataset):

    def __init__(self, data_path, new=True, target_column='diagnose', drop_column='Uitkomst - Behandeling', cutoff=40, top_x_targets=2,
                 to_merge_infrequent_targets=False, to_remove_infrequent_targets=True):
        super().__init__(data_path, target_column, cutoff, top_x_targets,
                         to_merge_infrequent_targets, to_remove_infrequent_targets)

        self.nominal_cols = ['heup_zijde', 'locatie_pijn']

        self.ordinal_cols = []

        self.binary_cols = ['wel-niet_traplopen', 'pijnstilling', 'type_pijnstilling__1', 'type_pijnstilling__2', 'type_pijnstilling__3', 'type_pijnstilling__4',
                            'fysiotherapie', 'werk', 'sport_hobby']

        self.cont_cols = ['duur', 'progressie', 'pijn_vs_bewegingsbeperking', 'sokken_schoenen', 'traplopen', 'startstijfheid', 'nachtpijn', 'VAS_rust', 'VAS_bewegen',
                          'werking_pijnstilling', 'effect_FT', 'invloed_klacht_werk', 'invloed_sport_hobby', 'leeftijd', 'lengte', 'gewicht', 'EQ5D', 'wens_operatie',
                          ]

        self.target_column = target_column
        self.drop_column = drop_column
        self.DATA_NAME = 'knee_ai'
        self.new = new

    @staticmethod
    def load_excel(path):
        return pd.read_excel(path, sheet_name='Data')

    def drop_other_cols(self):
        if self.new:
            old_prefixes = ['t0_ORT_', 't0_ORT_H_']
            self.learning_data.drop(
                [col for col in self.learning_data.columns for prefix in old_prefixes if prefix in col], axis=1, inplace=True)
        else:
            new_prefixes = ['t0_HAIn__H_', ' t0_HAIn__']
            self.learning_data.drop(
                [col for col in self.learning_data.columns for prefix in new_prefixes if prefix in col], axis=1, inplace=True)

    def preprocess(self):
        """Preprocess the learning data
        """
        self.drop_other_cols()

        # strip some unnecessary pre-and affixes from the variable names
        self.strip_from_column_names()

        # change some column types to prevent errors
        self.change_column_types()

        # remove rows with invalid data
        self.remove_invalid_target_rows()

        if self.to_merge_infrequent_targets:
            # merge targets that are infrequent to other
            self.merge_infrequent_targets()
        if self.to_remove_infrequent_targets:
            self.remove_infrequent_targets()

        # preprocess different data types
        self.process_ordinal()
        self.process_nominal()
        self.process_binary()
        self.process_continuous()

        # # remove rows with missing values
        # self.remove_bad_rows()

        # encode the labels to float
        self.encode_labels()
        # split the data in data and labels
        self.X, self.y = self.split_to_XY(self.learning_data)
        # self.feature_generation()

    def process_continuous(self):
        self.learning_data[self.cont_cols] = self.learning_data[self.cont_cols].replace(
            np.nan, -1)

    def change_column_types(self):
        """Change the types of some columns to prevent errors
        """
        # Changing column types
        self.learning_data[self.target_column] = self.learning_data[self.target_column].astype(
            str)

    def remove_invalid_target_rows(self):
        """Remove rows in learning data where the diagnosis is missing or contains an invalid character
        """
        # remove nans
        self.learning_data = self.learning_data[self.learning_data[self.target_column] != 'nan']

    def strip_from_column_names(self):
        """Strip unnecessary text from column names
        """
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('t0_ORT_', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('__antwoord_optie', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('_antwoord_optie', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('__antwoord_tekst', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('__antwoord_waarde', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('_antwoord_waarde', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('t0_HAIn__', '', x))
        self.learning_data = self.learning_data.rename(columns=lambda x: re.sub(
            'Uitkomst_1e_consult_-_Heup__Knie_uitkomst_arts__H_', '', x))
        self.learning_data = self.learning_data.rename(
            columns=lambda x: re.sub('H_', '', x))

    def process_binary(self):
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].astype(
            str)
        map_dict = {'Ja': 1, 'Nee': 0, '1.0': 0, '2.0': 1, '1': 0, '2': 1}
        self.learning_data[self.binary_cols] = self.learning_data[self.binary_cols].applymap(
            map_dict.get)

    def split_to_XY(self, data):
        data.drop_duplicates(subset='patientnummer', inplace=True)
        X = data.drop([self.target_column, self.drop_column, 'patientnummer', 'KL',
                       'type_werk', 'type_sport_hobby', 'poli_heup', 'meewerken_onderzoek', 'voorbeeld',
                       'VGOK_links',  'VGOK_welke_links',  'VGOK_rechts',  'VGOK_welke_rechts'], axis=1)
        y = data[self.target_column]

        missing_values = dict(self.learning_data.isnull().sum())
        print(missing_values)
        return X, y
