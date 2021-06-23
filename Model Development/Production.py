from __future__ import print_function

import sysconfig
from functools import partial
from pickle import load
from tkinter import *
from tkinter.filedialog import askopenfile, askopenfilename
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import shap
from PIL import Image, ImageTk

from data_classes import AiHipData, Dataset, KneeData, KneeDataPredict
from ml_classes import ModelDevelopment
from parameters import parameters as params


class Prediction:
    """Interface to predict knee diagnoses and treatments
    """

    def __init__(self, master):
        """Initialise and load the interface

        Args:
            master (TKinter Master): master of the interface
        """

        # define master
        self.master = master
        # white interface background
        self.master.configure(background="white", )
        # set the title of the interface
        master.title("St. Anna ML Project")
        # set the size of the interface
        master.geometry("450x900")
        # add the St. Anna logo at the top of the interface
        self.logo = ImageTk.PhotoImage(Image.open(
            'Images/st-anna-open-dag-1.png').resize((150, 100)))
        self.logo_label = Label(master, image=self.logo, pady=10)

        # retrieve data from database
        self.data_path = 'Data/ML data/knee_data_final.xlsx'

        #######################################################################################################################
        # add all buttons to the interface
        #######################################################################################################################

        # add button to browse files and select the file with the data to predict
        self.select_data_label = Label(
            self.master, text="Select Dataset:", bg='white', font=("sans-serif", 10), pady=10)
        self.data_button = Button(self.master, text='Browse', command=self.open_file,
                                  bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        # add button to preprocess the data for prediction of the diagnosis
        self.preprocess_diagnosis_label = Label(
            self.master, text="Click to prepare the diagnosis data:", bg='white', font=("sans-serif", 10), pady=10)
        self.preprocess_diagnosis_button = Button(self.master, text='Preprocess', command=self.preprocess_data_diagnosis,
                                                  bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        # add button to preprocess the data for prediction of the treatment category
        self.preprocess_treatment_label = Label(
            self.master, text="Click to prepare treatment the data:", bg='white', font=("sans-serif", 10), pady=10)
        self.preprocess_treatment_button = Button(self.master, text='Preprocess', command=self.preprocess_data_treatment,
                                                  bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        # add text field to enter the ID of the patient to predict
        self.id_label = Label(self.master, text="Fill in patient ID:",
                              bg='white', font=("sans-serif", 10), pady=10)
        self.patient_id_text = Entry(self.master, bd=4)

        # add button to predict the diagnosis of the patient
        self.predict_diagnosis_label = Label(
            self.master, text="Click to predict diagnosis:", bg='white', font=("sans-serif", 10), pady=10)
        self.predict_diagnosis_button = Button(self.master, text='Predict Diagnosis', command=self.predict_diagnosis,
                                               bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        # add button to predict the treatment of the patient
        self.predict_treatment_label = Label(
            self.master, text="Click to predict treatment:", bg='white', font=("sans-serif", 10), pady=10)
        self.predict_treatment_button = Button(self.master, text='Predict Treatment', command=self.predict_treatment_category,
                                               bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        # add button to explain the prediction of the diagnosis
        self.explain_diagnosis_label = Label(
            self.master, text="Click to explain diagnosis:", bg='white', font=("sans-serif", 10), pady=10)
        self.explain_diagnosis_button = Button(self.master, text='Explain diagnosis', command=lambda: self.explain_model(
            'diagnosis'), bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        # add button to explain the prediction of the treatment category
        self.explain_treatment_label = Label(
            self.master, text="Click to explain treatment:", bg='white', font=("sans-serif", 10), pady=10)
        self.explain_treatment_button = Button(self.master, text='Explain treatment', command=lambda: self.explain_model(
            'treatment'), bg='#0032A0', fg='white', font=("sans-serif", 10, "bold"), pady=6, padx=5)

        #######################################################################################################################
        # change colors of buttons when hovering over them
        #######################################################################################################################

        self.data_button.bind("<Enter>", lambda event, h=self.data_button: h.configure(
            bg="white", fg='#0032A0'))
        self.data_button.bind("<Leave>", lambda event, h=self.data_button: h.configure(
            bg="#0032A0", fg='white'))

        self.preprocess_diagnosis_button.bind(
            "<Enter>", lambda event, h=self.preprocess_diagnosis_button: h.configure(bg="white", fg='#0032A0'))
        self.preprocess_diagnosis_button.bind(
            "<Leave>", lambda event, h=self.preprocess_diagnosis_button: h.configure(bg="#0032A0", fg='white'))

        self.preprocess_treatment_button.bind(
            "<Enter>", lambda event, h=self.preprocess_treatment_button: h.configure(bg="white", fg='#0032A0'))
        self.preprocess_treatment_button.bind(
            "<Leave>", lambda event, h=self.preprocess_treatment_button: h.configure(bg="#0032A0", fg='white'))

        self.predict_diagnosis_button.bind(
            "<Enter>", lambda event, h=self.predict_diagnosis_button: h.configure(bg="white", fg='#0032A0'))
        self.predict_diagnosis_button.bind(
            "<Leave>", lambda event, h=self.predict_diagnosis_button: h.configure(bg="#0032A0", fg='white'))

        self.predict_treatment_button.bind(
            "<Enter>", lambda event, h=self.predict_treatment_button: h.configure(bg="white", fg='#0032A0'))
        self.predict_treatment_button.bind(
            "<Leave>", lambda event, h=self.predict_treatment_button: h.configure(bg="#0032A0", fg='white'))

        self.explain_diagnosis_button.bind(
            "<Enter>", lambda event, h=self.explain_diagnosis_button: h.configure(bg="white", fg='#0032A0'))
        self.explain_diagnosis_button.bind(
            "<Leave>", lambda event, h=self.explain_diagnosis_button: h.configure(bg="#0032A0", fg='white'))

        self.explain_treatment_button.bind(
            "<Enter>", lambda event, h=self.explain_treatment_button: h.configure(bg="white", fg='#0032A0'))
        self.explain_treatment_button.bind(
            "<Leave>", lambda event, h=self.explain_treatment_button: h.configure(bg="#0032A0", fg='white'))

        ##############################################################################################################
        # determine the placement of all the elements in the interface
        ##############################################################################################################

        self.select_data_label.grid(row=4, column=0, padx=5, pady=2,
                                    sticky=W+E+N+S)
        self.data_button.grid(row=5, column=0, padx=30,
                              pady=2,  sticky=W+E+N+S)
        self.id_label.grid(row=6, column=0, padx=30, pady=10,  sticky=W+E+N+S)
        self.patient_id_text.grid(
            row=7, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.preprocess_diagnosis_label.grid(
            row=8, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.preprocess_diagnosis_button.grid(
            row=9, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.preprocess_treatment_label.grid(
            row=8, column=1, padx=30, pady=2,  sticky=W+E+N+S)
        self.preprocess_treatment_button.grid(
            row=9, column=1, padx=30, pady=2,  sticky=W+E+N+S)
        self.predict_diagnosis_label.grid(
            row=10, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.predict_diagnosis_button.grid(
            row=11, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.predict_treatment_label.grid(
            row=10, column=1, padx=30, pady=2,  sticky=W+E+N+S)
        self.predict_treatment_button.grid(
            row=11, column=1, padx=30, pady=2,  sticky=W+E+N+S)

        self.explain_diagnosis_label.grid(
            row=12, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.explain_diagnosis_button.grid(
            row=13, column=0, padx=30, pady=2,  sticky=W+E+N+S)
        self.explain_treatment_label.grid(
            row=12, column=1, padx=30, pady=2,  sticky=W+E+N+S)
        self.explain_treatment_button.grid(
            row=13, column=1, padx=30, pady=2,  sticky=W+E+N+S)

        self.logo_label.grid(row=1, column=0, padx=30, columnspan=2, rowspan=1,
                             sticky=W+E+N+S, pady=5)

        self.text_box = Text(self.master, width=45, height=15, pady=10)
        self.text_box.grid(row=16, column=0, columnspan=2, pady=15)

        ##################################################################################################

    def open_file(self):
        """Function that is called when user clicks on the browse button.
           Loads the file that the user selects.
        """
        file = askopenfilename()
        self.data = pd.read_excel(file)

        self.message('Successfully loaded data')

    def preprocess_data_diagnosis(self):
        """Function that is called when the user clicks on the preprocess diagnosis button.
           Preprocesses the data for prediction of the diagnosis and selects the correct patient.
        """
        # instantiate the data class
        self.knee_data_diagnosis = KneeDataPredict(self.data)

        # preprocess the data
        self.knee_data_diagnosis.preprocess()

        # select the correct patient
        self.patient = self.knee_data_diagnosis.X.loc[self.knee_data_diagnosis.X['Nummer'].astype(
            str) == str(self.patient_id_text.get())].drop(['Nummer'], axis=1)

        # apply more preprocessing
        self.transform_data_diagnosis()

        # send message to interface
        self.message('Successfully preprocessed diagnosis data')

    def preprocess_data_treatment(self):
        """Function that is called when the user click on the preprocess treatment button.
           Preprocesses the data for prediction of the treatment category and selects the correct patient.
        """
        # instantiate the data class
        self.knee_data_treatment = KneeDataPredict(self.data)

        # preprocess the data
        self.knee_data_treatment.preprocess()

        # select the correct patient
        self.patient = self.knee_data_treatment.X.loc[self.knee_data_treatment.X['Nummer'].astype(
            str) == str(self.patient_id_text.get())].drop(['Nummer'], axis=1)

        # apply more preprocessing
        self.transform_data_treatment()

        # send message to interface
        self.message('Successfully preprocessed treatment data')

    def transform_data_diagnosis(self):
        """Apply more preprocessing
        """
        self.standardizer = load(
            open('Results/transformers/diagnosis_standardizer.pkl', 'rb'))
        self.patient = self.standardizer.transform(self.patient)

        self.poly = load(
            open('Results/transformers/diagnosis_polynomial_transformer.pkl', 'rb'))
        self.patient = self.poly.transform(self.patient)

    def transform_data_treatment(self):
        """Apply more preprocessing
        """
        self.poly = load(
            open('Results/transformers/polynomial_transformer_treatment.pkl', 'rb'))
        self.patient = self.poly.transform(self.patient)

    def predict_diagnosis(self):
        """Function that is called when user clicks on the predict diagnosis button.
           Predicts the diagnosis of the patient and reports the prediction to the interface.
        """
        # load the model
        self.diagnosis_model = load(
            open('Results/Models/gb_model_diagnosis.pkl', 'rb'))

        # predict the diagnosis of the patient
        diagnosis = self.diagnosis_model.predict(self.patient)

        # decode the diagnosis
        diagnosis = self.translate_diagnosis(diagnosis)

        # predict the probabilities
        odds = self.diagnosis_model.predict_proba(self.patient)

        # define the diagnosis labels
        labels = ['Anders', 'Gonartrose', 'Meniscusletsel']

        # generate the probabilities message
        odds = self.odds_to_str(odds, labels)

        # show the predicted diagnosis and probabilities on the interface
        self.message(f'Predicted Diagnosis: {diagnosis} \n Certainty: {odds}')
        return

    def predict_treatment_category(self):
        """Function that is called when user clicks on the predict treatment button.
           Predicts the treatment category of the patient and reports the prediction to the interface
        """
        # load the model
        self.treatment_model = load(
            open('Results/Models/rf_model_treatment.pkl', 'rb'))

        # predict the treatment category of the patient
        treatment = self.treatment_model.predict(self.patient)

        # decode the treatment category
        treatment = self.translate_treatment(treatment)

        # predict the probabilities
        odds = self.treatment_model.predict_proba(self.patient)

        # define the treatment category labels
        labels = ['Conservatief', 'Operatief']

        # generate the probabilities message
        odds = self.odds_to_str(odds, labels)

        # show the predicted diagnosis and probabilities on the interface
        self.message(f'Predicted Treatment: {treatment} \n Certainty: {odds}')
        return

    @staticmethod
    def translate_treatment(treatment):
        """Decode the numeric representation of the treatment category

        Args:
            treatment (int): treatment value

        Returns:
            str: the treatment category
        """
        if treatment == 0:
            return 'Conservatief'
        else:
            return 'Operatief'

    @staticmethod
    def translate_diagnosis(diagnosis):
        """Decode the numeric representation of the diagnosis

        Args:
            diagnosis (int): treatment value

        Returns:
            str: the treatment category
        """

        if diagnosis == 0:
            return 'Anders'
        elif diagnosis == 1:
            return 'Gonartrose'
        else:
            return 'Meniscusletsel'

    @staticmethod
    def odds_to_str(odds, labels):
        """Generates a string representing the predicted probabilities

        Args:
            odds (list): list with class probabilities
            labels (list): labels of the classes

        Returns:
            str: probability message
        """
        odds_string = ''
        for odd, label in zip(odds[0], labels):
            odds_string += f'{label} : {round(odd, 3)} \n'
        return odds_string

    def explain_model(self, model_type):
        """Explain the prediction by the model

        Args:
            model_type (str): diagnosis or treatment
        """

        train_data = pd.read_excel('Results/Transformers/train_data.xlsx')
        patient_data = pd.DataFrame(self.patient, columns=train_data.columns)

        if model_type == 'diagnosis':
            explainer = shap.KernelExplainer(
                self.diagnosis_model.predict_proba, train_data, link="logit")
        else:
            explainer = shap.KernelExplainer(
                self.treatment_model.predict_proba, train_data, link="logit")

        shap_values = explainer.shap_values(patient_data, nsamples=100)
        # shap_values = shap.TreeExplainer(self.diagnosis_model).shap_values(self.X)
        shap.summary_plot(shap_values, patient_data, plot_type="bar")
        plt.savefig(
            f'Results/Plots/SHAP/{self.patient_id_text.get()}_cm_summary.png')

        shap.plots.force(explainer.expected_value[0], shap_values[0])
        plt.savefig(
            f'Results/Plots/SHAP/{self.patient_id_text.get()}_cm_force.png')

        self.message('Explained model. Look in SHAP folder')

    def message(self, message):
        """Generates message to be sent to the interface"""
        # remove previous text from fields
        self.text_box.delete('1.0', END)
        # clear clipboard and add message to clipboard
        self.master.clipboard_clear()
        self.master.clipboard_append(message)
        self.text_box.insert(END, message)
        print(message)


root = Tk()
my_gui = Prediction(root)
root.mainloop()
