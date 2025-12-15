import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def ConfusionMatrix(model, X_test, y_test, le):
    class_labels = le.classes_

    label_mapping = {
        idx: tuple(label.split('_', 1))
        for idx, label in enumerate(class_labels)
    }

    y_pred_raw = model.predict(X_test)

    if isinstance(y_pred_raw, np.ndarray) and y_pred_raw.ndim == 2:
        y_pred = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred = np.asarray(y_pred_raw).astype(int)

    y_true = np.argmax(y_test, axis=1)

    true_emotions = [label_mapping[i][0] for i in y_true]
    true_intensities = [label_mapping[i][1] for i in y_true]
    pred_emotions = [label_mapping[i][0] for i in y_pred]
    pred_intensities = [label_mapping[i][1] for i in y_pred]

    results_df = pd.DataFrame({
        'Emotion attendue': true_emotions,
        'Emotion prédite': pred_emotions,
        'Intensité attendue': true_intensities,
        'Intensité prédite': pred_intensities
    })

    accuracy = np.mean(results_df["Emotion attendue"] == results_df["Emotion prédite"]) * 100

    print(f"Accuracy of model : {round(accuracy, ndigits=2)}%")

    cm_emotion = confusion_matrix(results_df["Emotion attendue"], results_df["Emotion prédite"])

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm_emotion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(results_df["Emotion attendue"]),
        yticklabels=np.unique(results_df["Emotion attendue"])
    )

    plt.xlabel("Émotions prédites", fontsize=12)
    plt.ylabel("Émotions attendues", fontsize=12)
    plt.tight_layout()
    plt.show()