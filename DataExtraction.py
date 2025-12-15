import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def add_noise(data, noise_level=0.035):
    noise_amp = noise_level * np.random.uniform() * np.amax(data)

    return data + noise_amp * np.random.normal(size=data.shape[0])

def add_pitch(data, sr=22050, n_steps=4):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def extract_features(data, sr=22050, n_mfcc=13):
    zcr = np.squeeze(librosa.feature.zero_crossing_rate(y=data))
    rmse = np.squeeze(librosa.feature.rms(y=data))
    mfcc = np.ravel(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc).T)
    # S = np.ravel(librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.max).T)

    return np.hstack((zcr, rmse, mfcc))
    # return np.hstack((zcr, rmse, mfcc, S))

def get_features(path, duration=2.5, offset=0.6, sr=None, DataAugmentation=True):
        data, sr = librosa.load(path, duration=duration, offset=offset, sr=None)
        audio = [extract_features(data, sr=sr)]

        if DataAugmentation:
            audio.append(extract_features(add_noise(data), sr=sr))
            audio.append(extract_features(add_pitch(data, sr=sr), sr=sr))
            audio.append(extract_features(add_noise(add_pitch(data, sr=sr)), sr=sr))

        return np.array(audio)

def process_feature(path, emotion, intensity, DataAugmentation=True):
        features = get_features(path, DataAugmentation=DataAugmentation)
        X_local = [f for f in features]
        Y_local = [(emotion, intensity)] * len(features)

        return X_local, Y_local

def DataExtraction(folderpath, DataAugmentation=True):
    files  = os.listdir(folderpath)[:]

    paths = []
    emotions = []
    intensities = []

    for file in files:
        parts = file.split("_")

        paths.append(os.path.join(folderpath, file))
        emotions.append(parts[2])
        
        try:
            intensities.append(parts[3][:2])
        except:
            intensities.append("NA")

    print(f"{len(paths)} files found")
    print(f"Extracting features...")

    X, Y = [], []

    for path, emotion, intensity in tqdm(zip(paths, emotions, intensities), total=len(paths)):
        x_local, y_local = process_feature(path, emotion, intensity, DataAugmentation=DataAugmentation)
        X.extend(x_local)
        Y.extend(y_local)

    print("Feature extraction completed.")

    chunk_size = 10
    dfs = []

    for i in tqdm(range(0, len(X), chunk_size), desc="Building DataFrame"):
        chunk_X = X[i:i+chunk_size]
        chunk_Y = Y[i:i+chunk_size]
        df_chunk = pd.DataFrame(chunk_X)
        df_chunk['Label'] = [f'{y[0]}_{y[1]}' for y in chunk_Y]
        dfs.append(df_chunk)

    features = dfs[0]

    for df in tqdm(dfs[1:], desc="Joining DataFrame "):
        features = pd.concat([features, df], ignore_index=True)

    print("Saving to features.csv...")

    features.to_csv('features.csv', index=False)

    print("features.csv saved.")
