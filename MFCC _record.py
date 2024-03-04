import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def get_mffc(audio_name):


    # Load the audio file
    audio_file = "records/"+audio_name  # Replace "your_audio_file.wav" with the path to your WAV file
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

def get_file_names_in_folder(folder_path):
    file_names = []
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the path is a file (not a folder)
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

get_mffc("hello_1.wav")


# Example usage:
folder_path = 'records'
file_names = get_file_names_in_folder(folder_path)
print(file_names)
for file in file_names:
    get_mffc(file)
