import pyaudio
import numpy as np

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

print("Listening...")

# Record audio
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# Close stream
stream.stop_stream()
stream.close()
audio.terminate()

# Convert frames to array
audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

# Calculate wavelength
wave_length = RATE / (np.argmax(audio_data[CHUNK:]) + CHUNK)

# Extract sample values from audio_data
sample_values = [
    np.mean(audio_data[:CHUNK]),  # Mean of the first CHUNK samples
    np.max(audio_data[CHUNK : 2 * CHUNK]),  # Max value of the second CHUNK samples
    np.std(
        audio_data[2 * CHUNK : 3 * CHUNK]
    ),  # Standard deviation of the third CHUNK samples
    np.min(audio_data[3 * CHUNK :]),  # Minimum value of the remaining samples
]

# Output the detected voice wavelength and features
print("Output:")
print(f"Detected voice features: {sample_values}")
