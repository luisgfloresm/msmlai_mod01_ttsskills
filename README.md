# msmlai_mod01_ttsskills
## Author and Purpose
Creator:
Luis G. Flores M.

Project for Master of Machine Learning and Artificial Intelligence
Module 01 - Fundaments of Programming (Python)
OBS Business School

Date: 10 May 2026

## Project Description
This project consists of a small demo application using Gradio and machine learning models to process audio and text.
The applications contains 3 main tabs:
* Tab 1 - Speech-to-Text and Sentiment Analyzer - The app takes an audio file or a recording using the microphone, and
  proceeds to generate a text transcription of the audio and analyze the sentiment of the text.
* Tab 2 - Text-to-Speech and Sentiment Analyzer - The app takes a text input and converts it to an audio file with a
  generated voice and also analyzes the sentiment of the text.
* Tab 3 - AI Meeting Assistant - This is a concept of a AI meeting assistant that records a meeting in audio format and
  then summarizes the text and extract key points. The app uses a Gemini API Key (user has to provide it) and an audio
  recording or file and proceeds to generate a text transcript, submits a prompt instruction to Gemini to summarize the
  text and extract key points in bulleted format and generates an audio file that provides the summary with a generated
  voice.

I have included some machine learning models available through Hugging Face. The machine learning models used in the 
application include the following:

* OpenAI's Whisper - openai/whisper-tiny
* DistilBERT - distilbert-base-uncased-finetuned-sst-2-english
* Meta's MMS - facebook/mms-tts-eng

I also included a feature functionality using Google Gemini with a Gemini API Key using Google's Gemini 2.5 Flash.

There is a live demo published in Hugging Face Spaces, through the following URL:
https://huggingface.co/spaces/lfloresmorales/ttsskills

## Requirements/Dependencies
In the requirements.txt file you may find the requirements needed for this application to work, including:
* gradio
* transformers
* torch
* google-generativeai
* librosa
* accelerate

Bash command that you may use to install dependencies:
'''bash
pip install gradio transformers torch google-generativeai librosa accelerate
'''

The user would need to generate a Gemini API Key through Google AI Studio, to be able to use the Tab 3 app.

## Screenshots of App in Hugging Face Spaces
Here I am including some screenshots of the app deployed in Hugging Face Spaces.

### Tab 1 - Speech-to-Text and Sentiment Analyzer
<img width="1988" height="1028" alt="Screenshot Tab 1 - Speech-to-Text and Sentiment Analyzer" src="https://github.com/user-attachments/assets/ff740ab3-b9a2-426b-9426-74dc02bc214e" />

### Tab 2 - Text-to-Speech and Sentiment Analyzer
<img width="2032" height="1072" alt="Screenshot Tab 2 -  Text-to-Speech and Sentiment Analyzer" src="https://github.com/user-attachments/assets/db8b2dee-f97b-4872-a92b-ec175edf174b" />

### Tab 3 - AI Meeting Assistant
<img width="1988" height="1118" alt="Screenshot Tab 3 - AI Meeting Assistant" src="https://github.com/user-attachments/assets/a08fd5b7-cdb1-44f8-a650-865131bae839" />
