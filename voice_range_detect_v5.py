import os
import numpy as np
from pydub import AudioSegment
import re
import glob

# Parameters
CHUNK = 1024


# Function to read an audio file
def read_mp3_file(filename):
    # Load the mp3 file
    audio = AudioSegment.from_mp3(filename)

    # Convert to mono
    audio = audio.set_channels(1)

    # Get frame rate (sample rate)
    framerate = audio.frame_rate

    # Get the raw audio data as a bytestring
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)

    return audio_data, framerate


# Function to process audio data
def process_audio_data(audio_data, framerate):
    # Calculate wavelength
    wave_length = framerate / (np.argmax(audio_data[CHUNK:]) + CHUNK)

    # Extract sample values from audio_data
    sample_values = [
        np.mean(audio_data[:CHUNK]),  # Mean of the first CHUNK samples
        np.max(audio_data[CHUNK : 2 * CHUNK]),  # Max value of the second CHUNK samples
        np.std(
            audio_data[2 * CHUNK : 3 * CHUNK]
        ),  # Standard deviation of the third CHUNK samples
        np.min(audio_data[3 * CHUNK :]),  # Minimum value of the remaining samples
    ]

    return wave_length, sample_values


# Main function to process all audio files in a folder
def main():
    # Directory containing the audio files
    folder_path = "voice/"

    # Verify folder path
    print("Folder path:", folder_path)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder not found.")
        return
    dataArray = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            # Read the audio file
            audio_data, framerate = read_mp3_file(file_path)

            # Process the audio data
            wave_length, sample_values = process_audio_data(audio_data, framerate)

            # Output the detected voice wavelength and features
            print(f"File: {filename}")

            match = re.search(r"\d+", filename)
            if match:
                age = match.group()
            print(age)

            # print(f"Detected voice wavelength: {wave_length}")

            # For Mor Efficent.
            # print(f"Detected voice features: {sample_values}")

            # dataArray = [age, wave_length]

            dataArray.append([age, wave_length])

    print("i'm all Data meeeee ===> ", dataArray)


# Easy Data
data1 = [[23, 0.2085747392815759]]


# Complex Data
# data = [
#     [-0.69553125, 115, 318.948724136786836, -6681, 23],
#     [-0.3953125, 115, 317.948724136786836, -6681, 285],
#     [-0.6953125, 115, 316.948724136786836, -6681, 23],
#     [-0.64953125, 115, 315.948724136786836, -6681, 285],
#     [-0.69553125, 115, 314.948724136786836, -6681, 286],
#     [-0.69653125, 115, 31.948724136786836, -6681, 287],
#     [-0.697563125, 115, 33.948724136786836, -6681, 289],
#     [-0.69533125, 115, 31.948724136786836, -6681, 280],
#     [-0.6953125, 115, 31.948724136786836, -6681, 284],
#     [-0.69531625, 115, 31.948724136786836, -6681, 283],
# ]

if __name__ == "__main__":
    main()
