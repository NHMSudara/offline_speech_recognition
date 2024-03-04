import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pyaudio
import numpy as np
import wave

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"  # Define the filename for the output WAV file



def rec_wav():
# Open stream
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
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


# Load the trained model
model = load_model('my_model.h5')  # Replace 'your_model.h5' with the path to your trained model file

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return None
    # Resize the image to match the input shape of the model
    image = cv2.resize(image, (170, 143))
    # Transpose the dimensions
    image = np.transpose(image, (1, 0, 2))
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    return image

# Function to predict label
def predict_label(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    if image is None:
        return None
    # Reshape the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    # Predict the label
    predictions = model.predict(image)
    print(predictions)
    # Get the predicted class (index with highest probability)
    predicted_class = np.argmax(predictions[0])
    print(predictions[0][predicted_class])
    if predicted_class != 2 and predictions[0][predicted_class]<0.33:
        return 2
    return predicted_class

def get_mffc(audio_name):
    # Load the audio file
    audio_file =audio_name  # Replace "your_audio_file.wav" with the path to your WAV file
    audio_data, sr = librosa.load(audio_file, sr=None)

    # Normalize the audio data
    audio_data /= np.max(np.abs(audio_data))  # Normalize to range [-1, 1]

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    # Get the size of the MFCCs plot
    num_frames = mfccs.shape[1]

    # Set the figure size to match the dimensions of the MFCCs plot
    plt.figure(figsize=(num_frames / 100, 2))

    # Plot MFCCs without axis ticks and labels
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)

    plt.axis('off')  # Turn off axis ticks and labels

    plt.tight_layout()

    # Create a new folder if it doesn't exist
    output_folder = "mfcc_plots"
    os.makedirs(output_folder, exist_ok=True)
    fig_name=audio_name.split(".")[0]+".png"
    # Save the plot in the new folder
    output_file = os.path.join(output_folder, fig_name)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)





while True:
    rec_wav()
    # Path to the new image
    wav_path = 'output.wav'  # Replace 'path_to_your_image.jpg' with the path to your image
    # Predict the label
    image_path="mfcc_plots/output.png"
    get_mffc(wav_path)
    predicted_label = predict_label(image_path)
    print("Predicted Label:", predicted_label)

    if predicted_label==1 :
        print("++++++++++++++ play music ++++++++++++++++++++")

    elif predicted_label==0:
        print("+++++++++++++++++ Hello ++++++++++++++++++++++")

    else:
        pass    

