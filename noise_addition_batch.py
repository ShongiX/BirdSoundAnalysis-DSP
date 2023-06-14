import os
import numpy as np
from scipy.io import wavfile

# Specify the folder containing the .wav files
folder_path = 'audio/wav/'

# Get a list of all .wav files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Define the noise spread to apply
spread = 0.5

# Loop through each file and apply white noise
for file_name in file_list:
    # Load the audio file
    file_path = os.path.join(folder_path, file_name)
    sample_rate, data = wavfile.read(file_path)

    # Normalize the audio data to the range [-1, 1]
    data = data.astype(np.float32) / 32767.0

    # Convert two channel audio into one channel
    data = np.mean(data, axis=1)

    # Generate white noise with the same length as the original audio
    noise = np.random.normal(0, spread, len(data))

    # Add the noise to the original audio
    noised_data = data + noise

    # Normalize the noised audio to ensure it stays within the [-1, 1] range
    max_value = np.max(np.abs(noised_data))
    noised_data /= max_value / 0.9  # Reducing the normalization slightly to prevent clipping

    # Convert the noised data back to 16-bit integer format
    output_data = (noised_data * 32767.0).astype(np.int16)

    # Save the noised audio as a new WAV file
    output_file = file_name.replace('.wav', '_noised.wav')
    output_file_path = os.path.join('audio/noised_batch/', output_file)
    wavfile.write(output_file_path, sample_rate, output_data)

    print("Noised sample saved as", output_file)
