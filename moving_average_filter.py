import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Specify the folder containing the .wav files
folder_path = 'audio/wav/'

# Get a list of all .wav files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Loop through each file and apply the moving average filter
for file_name in file_list:
    # Load the audio file
    file_path = os.path.join(folder_path, file_name)
    sample_rate, data = wavfile.read(file_path)

    # Convert audio data to floating point values between -1 and 1
    data = data.astype(np.float32) / 32767.0

    # Convert two channel audio into one channel
    data = np.mean(data, axis=1)

    # Define the window size for the moving average filter
    window_size = 20

    # Apply the moving average filter
    filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')

    # Convert the filtered data back to 16-bit integer format
    output_data = (filtered_data * 32767.0).astype(np.int16)

    # Save the filtered audio as a new WAV file
    output_file_path = os.path.join('audio/filtered/', file_name.replace('.wav', '_ma_filter.wav'))
    wavfile.write(output_file_path, sample_rate, output_data)

    # Plot the original and filtered signals
    time = np.arange(len(data)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.title('Original Signal: ' + file_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 16)
    plt.ylim(-1, 1)
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_data)
    plt.title('Filtered Signal: ' + file_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 16)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig('plots/' + file_name.replace('.wav', '_ma_filter.png'))
    plt.close()
