import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Specify the folder containing the .wav files
folder_path = 'audio/filtered/'

# Get a list of all .wav files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Define the noise spreads to apply
noise_spread = np.arange(0.01, 0.51, 0.01)

# Loop through each file and apply noise for each spread
for file_name in file_list:
    # The script could add noise to all samples, but I was curious about the already filtered one
    if file_name != "CommonBlackbird_XC605371_audition_filtered.wav":
        continue

    # Load the audio file
    file_path = os.path.join(folder_path, file_name)
    sample_rate, data = wavfile.read(file_path)

    # Normalize the audio data to the range [-1, 1]
    data = data.astype(np.float32) / 32767.0

    # Convert two channel audio into one channel
    data = np.mean(data, axis=1)

    # Generate noised samples for each noise spread
    for spread in noise_spread:

        # Generate white noise with the same length as the original audio
        noise = np.random.normal(0, spread, len(data))

        # Add the noise to the original audio
        noised_data = data + noise

        # Normalize the noised audio to ensure it stays within the [-1, 1] range
        max_value = np.max(np.abs(noised_data))
        noised_data /= max_value / 0.9  # Reducing the normalization slightly to prevent clipping

        # Calculate the difference between the original and noised audio
        difference = noised_data - data

        # Convert the noised data back to 16-bit integer format
        output_data = (noised_data * 32767.0).astype(np.int16)

        # Save the noised audio as a new WAV file
        output_file = file_name.replace('.wav', '_noised_' + '%.2f' % spread + '.wav')
        output_file_path = os.path.join('audio/noised/', output_file)
        wavfile.write(output_file_path, sample_rate, output_data)

        # Plot the original audio, noised audio, and difference
        time = np.arange(len(data)) / sample_rate

        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(time, data)
        plt.title('Original Signal: ' + file_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Spread')

        plt.subplot(3, 1, 2)
        plt.plot(time, noised_data)
        plt.title(f'Noised Signal (Spread: {spread:.2f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Spread')

        plt.subplot(3, 1, 3)
        plt.plot(time, difference)
        plt.title('Difference')
        plt.xlabel('Time (s)')
        plt.ylabel('Spread')

        plt.tight_layout()
        plt.savefig('plots/' + output_file.replace('.wav', '.png'))
        plt.close()

        print("Noised sample with spread " + '%.2f' % spread + " saved as " + output_file)
