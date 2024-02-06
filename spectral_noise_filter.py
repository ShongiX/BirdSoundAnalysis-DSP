import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

# Load the bird sound audio file
folder_path = 'audio/wav/'
file_name = 'CommonBlackbird_XC605371.wav'
audio, sr = librosa.load(folder_path + file_name)

# Split the audio into noise-only and speech-plus-noise sections
noise_duration = 1.0  # Duration of the noise-only segment (in seconds)
noise_audio = audio[int(1.5 * sr):int(1.5 * sr + noise_duration * sr)]
# noise_audio = audio[int(14.4 * sr):int(14.4 * sr + noise_duration * sr)]
speech_audio = audio[:]

# Compute the noise spectrum using a short-time Fourier transform (STFT)
n_fft = 2048  # FFT size
hop_length = 512  # Hop length for STFT
noise_stft = librosa.stft(noise_audio, n_fft=n_fft, hop_length=hop_length)
noise_mag = np.abs(noise_stft)
noise_phase = np.angle(noise_stft)

# Compute the speech-plus-noise spectrum
speech_stft = librosa.stft(speech_audio, n_fft=n_fft, hop_length=hop_length)
speech_mag = np.abs(speech_stft)
speech_phase = np.angle(speech_stft)

# Estimate the noise spectrum from the noise-only spectrum
estimated_noise_mag = np.mean(noise_mag, axis=1, keepdims=True)

# Perform spectral subtraction
denoised_mag = np.maximum(speech_mag - estimated_noise_mag, 0.0)

# Reconstruct the denoised signal from the magnitude and phase information
denoised_stft = denoised_mag * np.exp(1j * speech_phase)
denoised_audio = librosa.istft(denoised_stft, hop_length=hop_length)

# Save the denoised audio as a WAV file
wavfile.write('audio/filtered/' + file_name.replace('.wav', '_spec_filter2.wav'), sr, denoised_audio)

# Plot the original and filtered signals
time = np.arange(len(audio)) / sr
time2 = np.arange(len(denoised_audio)) / sr
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, audio)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 16)
plt.ylim(-1, 1)
plt.subplot(2, 1, 2)
plt.plot(time2, denoised_audio)
plt.title('Signal with spectral reduction')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 16)
plt.ylim(-1, 1)
plt.tight_layout()
plt.savefig('plots/' + file_name.replace('.wav', '_spec_filter.png'))
plt.close()
