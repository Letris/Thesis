# Thesis
Improving Patients Satisfaction with Orthopaedic Outpatient Consultations

This repository contains the code that I developed for my master thesis project in fulfillment of the Msc Data Science & Entrepreneurship.

The structure of the repository is as follows:

The main folder contains three subfolders: Model Development, Trial, and Thesis. The thesis subfolder contains my thesis and the figures presented in it. 
The Model Development subfulder contains all the code that was written for the first phase of the research: the development of Machine Learning models for the prediction of orthopaedic diagnoses and treatments. The Trial subfolder contains the jupyter notebook with code that I used to analyse the results of the second phase of the study: a randomised controlled trial at the St. Anna Hospital. 

Specifically, the folder contain the following files:

**Thesis**:
- Figures: the figures presented in the thesis
- Thesis.pdf: the thesis itself

**Model Development**
- data_classes.py: this file contains classes that I wrote for the initial preprocessing of the data
- ml_classes.py: this file contains a class to hold the preprocessed data and with functions to develop and evaluatge a range of ML models
- util_classes.py: this file contains utility classes and functions that are called within the ml_classes.py file
- parameters.py: this file contains a class that holds all the parameters for tuning different aspects of model training. These can be adapted to change the experiments. It also contains a function for internally optimizing the parameters automatically
- main.py: the file that was run to perform the experiments. The chosen experiment depends on parameters set in parameters.py
- Production.py: this file contains the code for the final prediction tool. Running the file will spawn a user interface in TKinter
-
**Trial**
- Analysis.ipynb: jupyter notebook with code that was used to perform analyses on the pilot data from the RCT
- Documents subfolder: documents related to the trial
- Timelines: the timelines that were developed for the educational eHealth tool


