# Fake Review Detection using NLP and Machine Learning
## 🌐 Live Demo

The project is deployed as an interactive web application using Streamlit.

🔗 **Try the live app here:**  
https://fake-review-detection-bykirti.streamlit.app/
## Overview

Online platforms often suffer from fake or misleading reviews that can manipulate customer decisions. This project builds a Natural Language Processing (NLP) based machine learning system to automatically classify reviews as genuine or fake (Computer generated).

The model processes textual data using TF-IDF feature extraction and applies multiple machine learning algorithms to identify patterns in fraudulent reviews.
##  Key Features
- Detects computer-generated reviews
- Provides prediction confidence score
- Uses NLP preprocessing and TF-IDF vectorization
- Trained using Logistic Regression
- Deployed as a Streamlit web application
## Project Workflow
1. Dataset Collection  
2. Data Preprocessing  
3. Feature Extraction (TF-IDF)  
4. Model Training  
5. Model Evaluation  
6. Model Saved (.pkl)  
7. Streamlit Web Application  
8. Deployment

## Dataset

The dataset was obtained from Kaggle.

Dataset Source:  
https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

Before training the model, the dataset was preprocessed to improve model performance. The preprocessing steps include:

- Converting text to lowercase
- Removing punctuation
- Removing stopwords
- Cleaning noisy characters

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Pickle

## Text Preprocessing Techniques

The following Natural Language Processing (NLP) techniques were applied to clean and prepare the textual data before training the machine learning models:

- Removing punctuation characters
- Transforming text to lowercase
- Eliminating stopwords
- Removing digits
- Stemming
- Lemmatization
 ##  Text Feature Extraction

To convert textual reviews into numerical features suitable for machine learning models, the following vectorization techniques were used:

- CountVectorizer (Bag of Words)
- TF-IDF Transformer (Term Frequency – Inverse Document Frequency)

##  Machine Learning Algorithms Used

The following classification algorithms were trained and evaluated to detect fake reviews:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier
- XGBoost Classifier

## Final Model Used

After evaluating multiple machine learning algorithms, **Logistic Regression** was selected as the final model due to its superior performance on the dataset.
#### Model Performance
The final Logistic Regression model achieved the following performance:

- **Accuracy:** 89.7%
- **Precision and recall are balanced**, indicating reliable detection of both fake and genuine reviews.



