import os
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def RunPCA(X_train, y_train, pca_model_path, n_components=75, random_state=42):
    print("Training PCA model...")

    y_train_int = np.argmax(y_train, axis=1)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_pca, y_train_int)

    os.makedirs(os.path.dirname(pca_model_path) or ".", exist_ok=True)
    joblib.dump({'pca': pca, 'rf': rf}, pca_model_path)

    print("PCA model training completed.")
    print(f"PCA model saved to {pca_model_path}.")