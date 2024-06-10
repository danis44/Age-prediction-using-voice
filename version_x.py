import numpy as np
import pyaudio

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10


# Load data from file
def load_data(filename):
    with open(filename, "r") as file:
        data = eval(file.read())  # Using eval to safely evaluate the list from the file
    return data


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


# Function to record audio for a specified duration
def record_audio(duration):
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print(f"Recording for {duration} seconds...")

    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert byte data to numpy array
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

    return audio_data, RATE


def find_closest_value(sample_values, data):
    closest_value = None
    min_distance = float("inf")

    for item in data:
        value, _ = item
        value = float(value)  # Convert value to float
        distance = np.linalg.norm(np.array(sample_values) - np.array(value))
        if distance < min_distance:
            min_distance = distance
            closest_value = item

    return closest_value


# Function to process audio provided by the user
def process_user_audio():
    # Load data from file
    filename = "data_v5.txt"
    data = load_data(filename)

    # Record audio for the specified duration
    audio_data, framerate = record_audio(RECORD_SECONDS)

    # Process the recorded audio data
    wave_length, sample_values = process_audio_data(audio_data, framerate)

    # Output the detected voice wavelength and features
    # print("Detected voice wavelength:", wave_length)
    # print("Detected voice features:", sample_values)

    # Find the closest value
    closest_value = find_closest_value(sample_values, data)

    # Print the closest value
    print("Closest Age:", closest_value[0])


# Call the function to process user-provided audio
process_user_audio()
