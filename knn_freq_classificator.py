import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the path to the audio files
data_path = 'audio/filtered/'

# Define the classes
classes = ['CommonBlackbird', 'WhiteStork']

# Initialize empty lists to store features and labels
features = []
labels = []

# Extract features from audio files
for filename in os.listdir(data_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(data_path, filename)
        waveform, sample_rate = librosa.load(file_path, sr=None)
        # Extract desired features from waveform
        feature_vector = librosa.feature.mfcc(y=waveform, sr=sample_rate)

        # Plot MFCC
        plt.figure()
        librosa.display.specshow(feature_vector, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.show()

        features.append(feature_vector.flatten())
        # Extract the class label from the filename
        label = filename.split('_')[0]
        labels.append(label)

# Convert feature and label lists to numpy arrays
max_length = max(len(lst) for lst in features)
features_padded = []
for lst in features:
    lst_array = np.array(lst).reshape(1, -1)  # Convert to (1, length)
    padding = np.zeros((1, max_length - lst_array.shape[1]))  # Pad with zeros
    padded_lst = np.concatenate((lst_array, padding), axis=1)  # Concatenate
    features_padded.append(padded_lst)

X = np.squeeze(np.array(features_padded))
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier for different k values
for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for k =", k, ":", accuracy)
