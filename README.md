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
```bash
pip install gradio transformers torch google-generativeai librosa accelerate
```

The user would need to generate a Gemini API Key through Google AI Studio, to be able to use the Tab 3 app.

## Screenshots of App in Hugging Face Spaces
Here I am including some screenshots of the app deployed in Hugging Face Spaces.

### Tab 1 - Speech-to-Text and Sentiment Analyzer
<img width="1988" height="1028" alt="Screenshot Tab 1 - Speech-to-Text and Sentiment Analyzer" src="https://github.com/user-attachments/assets/ff740ab3-b9a2-426b-9426-74dc02bc214e" />

### Tab 2 - Text-to-Speech and Sentiment Analyzer
<img width="2032" height="1072" alt="Screenshot Tab 2 -  Text-to-Speech and Sentiment Analyzer" src="https://github.com/user-attachments/assets/db8b2dee-f97b-4872-a92b-ec175edf174b" />

### Tab 3 - AI Meeting Assistant
<img width="1988" height="1118" alt="Screenshot Tab 3 - AI Meeting Assistant" src="https://github.com/user-attachments/assets/a08fd5b7-cdb1-44f8-a650-865131bae839" />


# msmlai_mod01_ttsskills - Español
## Autor and Propósito
Creador:
Luis G. Flores M.

Proyecto para Master of Machine Learning and Artificial Intelligence
Módulo 01 - Fundamentos de la Programación (Python)
OBS Business School

Fecha: 10 Mayo 2026

## Descripción del Proyecto
Este proyecto consiste en una aplicación de demostración pequeña que utiliza Gradio y modelos de aprendizaje automático 
(machine learning) para procesar audio y texto. La aplicación contiene 3 pestañas principales:
* Pestaña 1 - Analizador de Sentimiento y Voz a Texto (Speech-to-Text): La aplicación recibe un archivo de audio o una
  grabación mediante el micrófono, genera una transcripción de texto del audio y analiza el sentimiento del contenido.
* Pestaña 2 - Analizador de Sentimiento y Texto a Voz (Text-to-Speech): La aplicación recibe una entrada de texto, la
  convierte en un archivo de audio con una voz generada y también analiza el sentimiento del texto.
* Pestaña 3 - Asistente de Reuniones con IA: Este es un concepto de asistente de reuniones que graba una sesión en formato
  de audio para luego resumir el texto y extraer puntos clave. La aplicación utiliza una clave de API de Gemini (proporcionada
  por el usuario) y una grabación o archivo de audio; posteriormente, genera una transcripción, envía una instrucción (prompt)
  a Gemini para resumir el texto en formato de viñetas y genera un archivo de audio que narra dicho resumen con una voz
  generada.

He incluido varios modelos de aprendizaje automático disponibles a través de Hugging Face. Los modelos utilizados en la 
aplicación son:
* Whisper de OpenAI: openai/whisper-tiny
* DistilBERT: distilbert-base-uncased-finetuned-sst-2-english
* MMS de Meta: facebook/mms-tts-eng

También integré una funcionalidad que utiliza Google Gemini (específicamente el modelo Gemini 2.5 Flash) mediante una 
clave de API de Gemini.

Hay una demostración en vivo publicada en Hugging Face Spaces a través de la siguiente URL: 
https://huggingface.co/spaces/lfloresmorales/ttsskills

## Requerimientos / Dependencias
En el archivo requirements.txt podrás encontrar los requerimientos que se necesitan para que la aplicación funcione, 
incluyendo:
* gradio
* transformers
* torch
* google-generativeai
* librosa
* accelerate

Bash command para instalar las dependencias:
```bash
pip install gradio transformers torch google-generativeai librosa accelerate
```

El usuario tendrá que generar una clave de API de Gemini a través de Google AI Studio, para podr utilizar la pestaña 3 
de la aplicación.

## Capturas de pantalla de la App en Hugging Face Spaces
Aquí incluyo algunas capturas de pantalla de la app desplegada en Hugging Face Spaces.

### Tab 1 - Speech-to-Text and Sentiment Analyzer
<img width="1988" height="1028" alt="Screenshot Tab 1 - Speech-to-Text and Sentiment Analyzer" src="https://github.com/user-attachments/assets/ff740ab3-b9a2-426b-9426-74dc02bc214e" />

### Tab 2 - Text-to-Speech and Sentiment Analyzer
<img width="2032" height="1072" alt="Screenshot Tab 2 -  Text-to-Speech and Sentiment Analyzer" src="https://github.com/user-attachments/assets/db8b2dee-f97b-4872-a92b-ec175edf174b" />

### Tab 3 - AI Meeting Assistant
<img width="1988" height="1118" alt="Screenshot Tab 3 - AI Meeting Assistant" src="https://github.com/user-attachments/assets/a08fd5b7-cdb1-44f8-a650-865131bae839" />
