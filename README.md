# Spam Email Detection:

## Objective:
The objective of this project is to build a Machine Learning model that can automatically classify emails/messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP) techniques.
The system aims to improve email filtering by detecting unwanted or fraudulent messages accurately.

## Tools & Technologies Used:
Python
1.Pandas
2.NumPy 
3.Scikit-learn
4.TF-IDF Vectorizer 
5.Multinomial Naive Bayes
6.Matplotlib & Seaborn

## Steps Used in the Project:
1.Import Libraries
Imported necessary Python libraries for data processing, visualization, and modeling.

2.Load Dataset
Loaded the spam dataset and selected relevant columns (label and message).

3.Data Preprocessing
Renamed columns for clarity
Converted labels (ham → 0, spam → 1)

4.Text Vectorization
Applied TF-IDF Vectorizer to convert text messages into numerical features.
Train-Test Split
Split the dataset into training (80%) and testing (20%) sets.

5.Model Training
Trained the model using Multinomial Naive Bayes algorithm.

6.Model Evaluation
Evaluated performance using:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
Visualization
Plotted confusion matrix using a heatmap.
Prediction on New Data
Tested the model using a custom sample email to classify it as spam or not spam.

## Outcome:
Successfully built a Spam Detection Model with high accuracy.
The model can effectively classify new emails as Spam or Not Spam.

TF-IDF combined with Multinomial Naive Bayes proved to be efficient for text classification tasks.

The project demonstrates practical implementation of NLP and Machine Learning concepts.
