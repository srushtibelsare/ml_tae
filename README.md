Personality Prediction Using the MBTI Dataset
Project Overview
This project builds an automated personality prediction system based on the Myers-Briggs Type Indicator (MBTI) framework. It analyses textual social media posts from users and classifies them into one of 16 MBTI personality types using machine learning techniques.
The entire pipeline — from raw text to predictions — was implemented and executed on Google Collab using Python.

Repository Structure
MBTI-Personality-Prediction/
|
+-- mbti_1.csv                        # Dataset (8,675 records)
|
+-- Notebooks (Google Collab .ipynb)
|   +-- Logistic_Regression70_30.ipynb
|   +-- Logistic_Regression80_20.ipynb
|   +-- Decision_Tree70_30.ipynb
|   +-- Decision_Tree80_20.ipynb
|   +-- Random_Forest70_30__1_.ipynb
|   +-- Random_Forest80_20.ipynb
|   +-- KNN_70_30_.ipynb
|   +-- KNN_80_20_.ipynb
|   +-- Bayes_70_30_.ipynb
|   +-- Bayes_80_20_.ipynb
|   +-- SVC_70_30_.ipynb
|   +-- SVC_80_20_.ipynb
|
+-- ml_finalereport.docx              # Final Project Report

Notebook Naming Convention: <ModelName><TrainTestSplit>.ipynb
Example: Logistic_Regression70_30.ipynb → Logistic Regression with 70% train / 30% test split
Dataset

Property	Details
File	mbti_1.csv
Records	8,675 rows
Columns	type (MBTI label),  posts (text)
Classes	16 MBTI personality types
Source	Publicly available MBTI dataset
Missing Data	None

16 MBTI Personality Types
INFJ  INFP  INTJ  INTP  ISFJ  ISFP  ISTJ  ISTP
ENFJ  ENFP  ENTJ  ENTP  ESFJ  ESFP  ESTJ  ESTP

Class Imbalance Note: Types like INFP are heavily overrepresented, while types like ESTJ have very few samples. This significantly impacts model accuracy.

Methodology
Pipeline Overview
Raw Text Data
     |
     v
Text Preprocessing (re library)
  - Remove URLs, hyperlinks
  - Remove MBTI keywords (to prevent data leakage)
  - Remove special characters, punctuation, numbers
  - Lowercase conversion
     |
     v
Feature Extraction
  - TF-IDF Vectorization (TfidfVectorizer)
     |
     v
Train-Test Split
  - 70:30 split
  - 80:20 split
     |
     v
Model Training & Evaluation
  - 6 ML Models x 2 Splits = 12 Notebooks
     |
     v
Performance Comparison
  - Accuracy, Precision, Recall, F1-Score

Models Implemented

#	Model	Notebook Files
1	Logistic Regression	Logistic_Regression70_30.ipynb,  Logistic_Regression80_20.ipynb
2	Decision Tree	Decision_Tree70_30.ipynb,  Decision_Tree80_20.ipynb
3	Random Forest	Random_Forest70_30__1_.ipynb,  Random_Forest80_20.ipynb
4	K-Nearest Neighbors (KNN)	KNN_70_30_.ipynb,  KNN_80_20_.ipynb
5	Naive Bayes (GaussianNB)	Bayes_70_30_.ipynb,  Bayes_80_20_.ipynb
6	Support Vector Machine	SVC_70_30_.ipynb,  SVC_80_20_.ipynb


Results
Table 1: 70:30 Train-Test Split

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.36	0.29	0.36	0.31
Decision Tree	0.26	0.17	0.26	0.17
Random Forest	0.23	0.17	0.23	0.12
SVM  ★	0.37	0.36	0.37	0.32
KNN	0.20	0.20	0.20	0.15
Naive Bayes	0.22	0.29	0.22	0.24



Table 2: 80:20 Train-Test Split

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.36	0.29	0.36	0.31
Decision Tree	0.26	0.17	0.26	0.17
Random Forest	0.23	0.17	0.23	0.12
SVM  ★	0.37	0.36	0.37	0.32
KNN	0.20	0.20	0.20	0.15
Naive Bayes	0.22	0.29	0.22	0.24

Key Findings
•	Best Model: SVM achieved highest accuracy (~37%) in 80:20 split
•	Best Consistent Model: Logistic Regression — stable at 36% across both splits
•	Worst Performing: KNN — lowest accuracy, precision, recall, and F1
•	Both splits produced nearly identical results, indicating model stability
•	Linear models (SVM, Logistic Regression) outperform tree-based and distance-based models on TF-IDF features

Technologies & Libraries

Category	Tools / Libraries
Platform	Google Colab
Language	Python 3
Data	pandas, numpy
NLP	re (regex), sklearn.TfidfVectorizer
ML Models	scikit-learn
Tuning	GridSearchCV
Visualization	matplotlib, seaborn

How to Run
Step 1: Clone / Download
Download all .ipynb files and the mbti_1.csv dataset.
Step 2: Upload to Google Drive
Upload mbti_1.csv to your Google Drive:
/content/drive/MyDrive/ml/mbti_1.csv
Step 3: Open in Google Colab
Open any .ipynb file in Google Colab.
Step 4: Mount Google Drive
Each notebook contains this cell — run it first:
from google.colab import drive
drive.mount('/content/drive')
Step 5: Run All Cells
Run all cells top to bottom. The notebook will:
1.	Load and explore the dataset
2.	Preprocess text
3.	Apply TF-IDF vectorization
4.	Split data (70:30 or 80:20)
5.	Train the model
6.	Evaluate and display metrics + plots

Evaluation Metrics

Metric	Description
Accuracy	Overall correct predictions / total predictions
Precision	Correct positive predictions / total predicted positives
Recall	Correct positive predictions / total actual positives
F1-Score	Harmonic mean of Precision and Recall

Challenges Faced
•	Noisy Text Data —  URLs, special characters, MBTI keywords required careful cleaning
•	Class Imbalance —  Some personality types had very few samples, biasing model predictions
•	High-Dimensional Features —  TF-IDF creates sparse, high-dimensional feature matrices
•	Low Accuracy Ceiling —  16-class classification with limited data creates a hard accuracy ceiling
•	Overfitting in Tree Models —  Decision Tree and Random Forest showed signs of overfitting

Team Members

USN	Name
CS23164	Sonal Ambhore
CS23165	Srushti Belsare

