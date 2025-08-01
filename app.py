import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_FOLDER = "audio"
SPECTROGRAM_FOLDER = "spectrograms"


os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)


for filename in os.listdir(AUDIO_FOLDER):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        filepath = os.path.join(AUDIO_FOLDER, filename)
        print(f"Processing {filename}...")

        
        y, sr = librosa.load(filepath)
        y = y[:sr * 10] 

        # Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Save as image
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')  # Hide axes
        image_name = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(SPECTROGRAM_FOLDER, image_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved: {save_path}")
