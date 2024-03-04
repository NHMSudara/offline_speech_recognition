import pyaudio
import numpy as np
import wave

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "records/noice_1.wav"  # Define the filename for the output WAV file

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

# Save recorded audio as WAV file
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_data))

print("Audio saved as", WAVE_OUTPUT_FILENAME)
