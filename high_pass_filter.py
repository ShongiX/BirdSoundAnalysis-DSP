import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Specify the folder containing the .wav files
folder_path = 'audio/wav/'

# Get a list of all .wav files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Loop through each file and apply the high-pass filter
for file_name in file_list:
    # Load the audio file
    file_path = os.path.join(folder_path, file_name)
    sample_rate, data = wavfile.read(file_path)

    # Convert audio data to floating point values between -1 and 1
    data = data.astype(np.float32) / 32767.0

    # Convert two channel audio into one channel
    data = np.mean(data, axis=1)

    # Define the cutoff frequency for the high-pass filter (in Hz)
    cutoff_frequency = 1000

    # Normalize the cutoff frequency
    nyquist = 0.5 * sample_rate
    cutoff_normalized = cutoff_frequency / nyquist

    # Create high-pass filter coefficients using a Butterworth filter
    b, a = butter(4, cutoff_normalized, btype='high')

    # Apply the high-pass filter using filtfilt
    filtered_data = filtfilt(b, a, data)

    # Convert the filtered data back to 16-bit integer format
    output_data = (filtered_data * 32767.0).astype(np.int16)

    # Save the filtered audio as a new WAV file
    output_file_path = os.path.join('audio/filtered/', file_name.replace('.wav', '_high_pass.wav'))
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
    plt.title('High-pass filtered Signal: ' + file_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 16)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig('plots/' + file_name.replace('.wav', '_high_pass.png'))
    plt.close()
