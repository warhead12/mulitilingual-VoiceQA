import pygame

def play_audio_with_pygame(audio_file_path):
    try:
        # Initialize the pygame library
        pygame.init()

        # Create a pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(audio_file_path)

        # Play the audio file
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            continue

        # Clean up and quit pygame
        pygame.quit()
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
audio_file = "C:/Users/panka/Desktop/hackout_2023/marathi_1_recorded_audio.flac"
play_audio_with_pygame(audio_file)
