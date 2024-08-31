# READ ME FILE #

This is your README. 

Write your name on line 6, save it, and then head back to GitHub Desktop.
HARIDOSS ANNAVI


**Credit Card Fault Prediction
Overview**
This project aims to build a machine learning model to predict credit card faults, such as defaults or fraudulent transactions. The goal is to assist financial institutions in identifying high-risk transactions or customers, thereby reducing losses and enhancing customer security.

Table of Contents
1.	Project Structure
2.	Dataset
3.	Dependencies
4.	Installation
5.	Usage
6.	Model Training
7.	Results
 
**Project Structure**

credit-card-fault-prediction/
│
├── data/
│   ├── HA_creditcard.csv         # The dataset
│   
│
├── notebooks/
│   └── ha-credit-card-fraud-prediction-rf-smote.ipynb # Jupyter notebook for project script
│
├── models/
│   └──  None # Trained KNN model
│
├── scripts/
│   ├── preprocess.py                # Script for data preprocessing
│   ├── train_model.py               # Script for training the model
│   └── evaluate_model.py            # Script for evaluating model performance
│
├── README.md                        # Project documentation

**Dependencies**
The project requires the following Python libraries:
•	pandas
•	numpy
•	scikit-learn
•	matplotlib
•	seaborn
•	jupyter

**GitHub Location**

https://github.com/haridoss-annavi/desktop-tutorial/blob/master/ha-credit-card-fraud-prediction-rf-smote.ipynb

**Model Training**

The project uses the following machine learning models to predict credit card faults:
•	K-Nearest Neighbors (KNN)
•	Logistic Regression
•	Random Forest Classifier
The best-performing model is chosen based on the accuracy, precision, recall, and F1 score.


**Results**
•	The final model achieved an accuracy of 
The highest values of Normal transactions are 284315, while of Fraudulent transactions are just 492.

The average value of normal transactions are small(USD 88.29) than fraudulent transactions that is USD 122.21

**Best score:**

SMOTE (OverSampling) = RandomForest =

Accuracy: 0.9995611109160493 Precision: 0.9041095890410958 Recall: 0.7857142857142857 F2: 0.806845965770171

This is a considerably difference by the second best model that is 0.8252 that uses just RandomForests with some Hyper Parameters.

**Worst Score: **

Logistic Regression with GridSearchCV to get the Best params to fit and predict where the recall = 66.67% and f2 = 70%.
