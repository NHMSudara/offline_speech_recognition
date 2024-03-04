import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

# Initialize empty list to store audio data
audio_data = []

# Record audio
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    decoded_data = np.frombuffer(data, dtype=np.int16)
    audio_data.extend(decoded_data)

print("Recording finished.")

# Close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Convert audio data to numpy array and normalize
audio_data = np.array(audio_data, dtype=np.float32)
audio_data /= np.max(np.abs(audio_data))  # Normalize to range [-1, 1]

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)

# Plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()


