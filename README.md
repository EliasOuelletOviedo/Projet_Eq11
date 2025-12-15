Analysis of CREMA-D dataset. 
Emotion recognition and classification

CREMA-D dataset : https://www.kaggle.com/datasets/ejlok1/cremad

Download and adjust path in main.py

Data extraction will result in a .csv file in "/RESULTS". Once data is extracted, "reset_extraction" should be set to "False" to accelerate process.

Model training will result in a .keras or .joblib file in "/MODELS". Once models are trained, they can be called back without retraining by setting "reset_CNN" or "reset_PCA" to "False".
