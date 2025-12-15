import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from CNNModel import RunCNN
from PCAModel import RunPCA
from ConfusionMatrix import ConfusionMatrix
from DataExtraction import DataExtraction

if not sys.warnoptions:
    warnings.simplefilter("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 

folderpath = "./CREMA-D/AudioWAV/"
features_path = "FEATURES/features.csv"
cnn_model_path = "MODELS/CNNModel.keras"
pca_model_path = "MODELS/PCAModel.joblib"

DataAugmentation = True

CNNModel = True
PCAModel = True

reset_extraction = True
reset_CNN = True
reset_PCA = True

if not(os.path.exists(features_path)) or reset_extraction:
    DataExtraction(folderpath, DataAugmentation=DataAugmentation)

print("Preparing data for training...")

features = pd.read_csv(features_path)
features=features.fillna(0)

X = features.drop(columns=["Label"]).values
y_to_encode = features["Label"].values

le = LabelEncoder()
y = to_categorical(le.fit_transform(y_to_encode))

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

print("Data preparation completed.")

if CNNModel:
    X_traincnn = np.expand_dims(X_train, axis=2)
    X_testcnn  = np.expand_dims(X_test, axis=2)

    if not(os.path.exists(cnn_model_path)) or reset_CNN:
        RunCNN(X_traincnn, X_testcnn, y_train, y_test, cnn_model_path)

    cnn_model = tf.keras.models.load_model(cnn_model_path)

    ConfusionMatrix(cnn_model, X_testcnn, y_test, le)

if PCAModel:
    if not(os.path.exists(pca_model_path)) or reset_PCA:
        RunPCA(X_train, y_train, pca_model_path)

    pca_model = joblib.load(pca_model_path)
    pca = pca_model['pca']
    rf = pca_model['rf']

    X_test_pca = pca.transform(X_test)

    ConfusionMatrix(rf, X_test_pca, y_test, le)
