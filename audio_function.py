import sounddevice as sd
import soundfile as sf
import numpy as np

def record_and_save_audio(sample_rate, duration, output_path):
    # Record audio
    print("Recording audio...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()

    # Save the audio data to the specified output path
    sf.write(output_path, audio_data, sample_rate)

    return output_path
