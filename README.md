


# Multilingual Voice Q&A

We are here basically making the multilingual speech to speech Question Answering sysetm, Here we have first taken the Audio file as input from the user in any language like Hindi, English, marathi, bangali,Tamil or any other

*Here is overview of our project*
![Screenshot 2024-03-14 113545](https://github.com/pankaj10032/mulitilingual-VoiceQA/assets/97507747/bf1092ec-169b-4de7-ba9e-0291c2bfcb9b)

## Installation

clone this repo [pip](https://github.com/pankaj10032/hackout2023_projects) to install foobar.

```bash
git clone https://github.com/pankaj10032/hackout2023_projects.git
```
Since I was running the code on server that's why I have no option of collection audio so I have collected audio manually by running the file , make sure to specify the path of audio in the **output_filename** otherwise just provide filename so it will save the Audio file in the same directory in which audio.py is present, make sure that format of the Audio file should be ".flac".

```bash
python audio.py
```
After running Audio.py Open the file name llama2_text_2_text.py", specify model_id you want like higher versions of llama, I have here used the llama7b chat model taken from the huggingface, also make sure to specify the huggingface token of your account on which you have the access of llama model, in the variable named 
 **hf_auth**


After that jusy get the path of that file 
## how to run the code
*First run the *
open the file  **config_hi.py**
# Description

## Audio Extraction:-
we are here basically making the multilingual speech to speech Question Answering sysetm, Here we have first taken the Audio file as input from the user in any language like Hindi, English, marathi, bangali,Tamil or any other foreign language and taking audio into ".flac" format since our openai/whisper-large-v2 support this format only with the frequency should be 16000HZ and time duration to be less like around 10 second or lesser, so that context widnow of the model doesn't get fit.

for running Audio extraction run the file named "audio.py" but make sure to specify the file path necessarily
Please make sure to update tests as appropriate.

## Audio(any language) to English text:-
After that we will convert our audio into English text so that it can be easily processed by LLM, using the "openai/whisper-large-v2" model using the transformers pipeline text generation since our task is just audio to text so need to fine tune the model. *for converting audio to text run the file named "audio_cleaning.py"

## Text to Text Question Answering:-
In this case we have user query as obtained from the last step from audio to text we take llama for text generation model using huggingface pipeiline, we have also a pdf(data) containing some information related to the railway we first load that data, split that data into chunk size of 1024 with chunk overlap of 20, then compute embeddings of all those chunk using sentence transformers models and store chunks of fixed size in the FAISS vector store, then using the conversation chain we just get the most relevant context among all those chunk(1024) size and passing them to LLM(llama for text generation) to get the relevant answer.
## make sure before using llama you must have the access of that, you can get the access of that on meta website by filling the google form of that and also get the access from huggingface also , put the samer Email id at both the places.

Here we can also use OpenAI but On the industry level that will be cosly,we can also use higher versions of Llama like 13b, 70b parameters model to get the better results.

## English text to Desire language by user(multilingual) text conversion:-
Here we have our answer as English text so we wlll convert that into multilingual language(marathi, hindi, bengali) as per desired by user

## Text to Audio conversion:-
Since in the last step we have got the text into mulitilingual languages that user want, Now we have converted that text into Audio using the google translator(gtts) library, since there is not any good mulitilingual(hindi, marathi or foreign languages), into Audio of the same language, but make sure to specify the language code like for the hindi code is 'hi' .


## problem solved by our model:-
The main use case of this mulitilingual Voice QA is st the Railway station, where 
