import os
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

# Define the folder containing MP3 files
voice_folder = "voice"


# Define a function to extract features from an audio segment
def extract_features(audio_segment):
    # Convert audio segment to raw audio data
    raw_data = np.array(audio_segment.get_array_of_samples())

    # Define chunk size
    chunk_size = len(raw_data) // 4

    # Calculate features
    features = [
        np.mean(raw_data[:chunk_size]),  # Mean of the first chunk
        np.max(raw_data[chunk_size : 2 * chunk_size]),  # Max of the second chunk
        np.std(raw_data[2 * chunk_size : 3 * chunk_size]),  # Std of the third chunk
        np.min,
    ]
