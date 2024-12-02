import sounddevice as sd
import soundfile as sf
import numpy as np

# Set the audio recording parameters
sample_rate = 16000  # Sample rate in Hz
duration = 10  # Recording duration in seconds
output_filename = "marathi_2_recorded_audio.flac"

# Record audio
print("Recording audio...")
audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
sd.wait()

# Save the recorded audio in FLAC format
sf.write(output_filename, audio_data, sample_rate)

print(f"Audio recorded and saved as {output_filename}")

