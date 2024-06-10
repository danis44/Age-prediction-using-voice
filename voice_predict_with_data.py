import pyaudio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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


# Placeholder for actual feature extraction from audio_data
def extract_features(audio_data, rate):
    # Example feature: average power spectrum density
    power_spectrum = np.abs(np.fft.fft(audio_data)) ** 2
    freqs = np.fft.fftfreq(len(audio_data), 1 / rate)
    return [np.mean(power_spectrum), np.std(power_spectrum), np.max(power_spectrum)]


# Simulated dataset with features extracted from real audio samples
dataset = [
    (extract_features(audio_data, RATE), 25),  # Example voice sample with age 25
    (extract_features(audio_data, RATE), 35),  # Example voice sample with age 35
    (extract_features(audio_data, RATE), 45),  # Example voice sample with age 45
]

# Extract features and target
X = np.array([sample[0] for sample in dataset])  # Extract features from dataset
y = np.array([sample[1] for sample in dataset])  # Extract target values from dataset

# Split the dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train a random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {val_mae:.2f}")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {test_mae:.2f}")

# Predict age based on detected voice features
voice_features = scaler.transform(
    np.array(extract_features(audio_data, RATE)).reshape(1, -1)
)
predicted_age = model.predict(voice_features)[0]
print("Predicted age:", predicted_age)
