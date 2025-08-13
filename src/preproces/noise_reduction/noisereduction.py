import os

import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


def noise_reduction(input_folder, output_folder):
    """Apply noise reduction to all MP3 files in the input folder and save to output folder."""
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            temp_wav = "temp.wav"
            temp_reduced_wav = "temp_reduced.wav"

            # Convert MP3 to WAV
            audio = AudioSegment.from_file(input_file, format="mp3")
            audio.export(temp_wav, format="wav")

            # Read WAV and convert to mono if needed
            rate, data = wavfile.read(temp_wav)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1).astype(data.dtype)

            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=data, sr=rate)

            # Save reduced audio
            wavfile.write(temp_reduced_wav, rate, reduced_noise)

            # Convert back to MP3
            final_audio = AudioSegment.from_file(temp_reduced_wav, format="wav")
            final_audio.export(output_file, format="mp3")

            # Cleanup
            os.remove(temp_wav)
            os.remove(temp_reduced_wav)

            print(f"Noise-reduced MP3 saved as {output_file}")


folders = [
    (
        "scr/data/original_data/BRY/MP3",
        "scr/data/noise_free_data/BRY/MP3",
    ),
    (
        "scr/data/original_data/CAL/MP3",
        "scr/data/noise_free_data/CAL/MP3",
    ),
]

if __name__ == "__main__":
    # Apply noise reduction to each folder
    for input_folder, output_folder in folders:
        noise_reduction(input_folder, output_folder)
