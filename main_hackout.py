import json
from audio_function import record_and_save_audio
from audio_cleaning import transcribe_audio
from llama2_text_2_text import setup_and_run_llm
from mbart_translator import translate_text
from text_to_speech_in_any_language import generate_audio
from Audio_play import play_audio_with_pygame
import config_hi
from config_hi import config
# Load parameters from the configuration file
# with open("config.json", "r") as config_file:
  #   config = json.load(config_file)

# Audio settings
audio_settings = config["audio_settings"]
# output_path = record_and_save_audio(audio_settings["sample_rate"], audio_settings["duration"], audio_settings["output_path"])
output_path=config["audio_settings"]["output_path"]
# language code in which we are taking input
language_input=audio_settings["language_input"]
# Language model settings
llm_settings = config["llm_settings"]
model_id = llm_settings["model_id"]

# query = config["query"]

# Translation settings
translation_settings = config["translation_settings"]
output_file_path = translation_settings["output_file_path"]
target_language_code = translation_settings["target_language_code"]


# Continue with the rest of your code as before
transcription = transcribe_audio(output_path, language_input)
answer = setup_and_run_llm(transcription)
translated_text = translate_text(answer, target_language_code)
generate_audio(translated_text, output_file_path)

