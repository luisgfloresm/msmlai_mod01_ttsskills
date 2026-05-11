import gradio as gr
from transformers import pipeline
import torch
import google.generativeai as genai

# Section 1 - Initialize the Pipelines from Hugging Face
# Whisper is excellent for transcription; DistilBERT is perfect for fast sentiment analysis
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# TTS Model: Meta's MMS (Massively Multilingual Speech) - Fast and high quality
tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# Section 2 - Define the Functions that will be used in the App
# Process Audio Function: Transcribes and analyzes audio input using Hugging Face pipelines
def process_audio(audio_path: str) -> tuple[str, str]:
    '''
    This function takes an audio file path as input, transcribes it using Whisper, 
    and analyzes its sentiment using DistilBERT.
    Args:
        audio_path (str): The path to the audio file.
    Returns:
        tuple: A tuple containing the transcription and sentiment analysis.
    '''
    # Check if the audio was provided, if not, return an error message
    if audio_path is None:
        return "No audio detected", "N/A"
    
    try:
        # Whisper handles the audio file path directly
        transcription = stt_pipeline(audio_path)["text"]
        
        # Generate the Sentiment Analysis
        sentiment_result = sentiment_pipeline(transcription)[0]
        label = sentiment_result['label']
        score = round(sentiment_result['score'], 4)
        sentiment_output = f"Analysis: {label} ({score:.2%} confidence)"
    except Exception as e:
        return f"Error: {str(e)}", "Error"
    
    # Return the transcription and sentiment analysis
    return transcription, sentiment_output

# Analyze and Speak Function: Converts text to speech and analyzes its sentiment using Hugging Face pipelines
def analyze_and_speak(text: str):
    '''
    This function takes a text as input, converts it to speech using MMS, 
    and analyzes its sentiment using DistilBERT.
    Args:
        text (str): The text to convert to speech and analyze.
    Returns:
        tuple: A tuple containing the audio data and sentiment analysis.
    '''
    # Check if the text was provided, if not, return an error message
    if not text.strip():
        return None, "Please enter some text."

    try:
        # Generate the Sentiment Analysis
        sentiment_result = sentiment_pipeline(text)[0]
        label = sentiment_result['label']
        score = sentiment_result['score']
        sentiment_output = f"Analysis: {label} ({score:.2%} confidence)"

        # Generate Audio (TTS)
        # The pipeline returns a dictionary with 'audio' (numpy array) and 'sampling_rate'
        audio_data = tts_pipeline(text)
    except Exception as e:
        return None, f"Error: {str(e)}"
    
    # Gradio gr.Audio component expects a tuple of (sample_rate, data)
    # Return the audio data and sentiment analysis
    return (audio_data["sampling_rate"], audio_data["audio"]), sentiment_output

# Process Meeting Function: Transcribes, summarizes, and speaks the meeting transcript using Hugging Face pipelines and Google Gemini
def process_meeting(api_key: str, audio_path: str):
    '''
    This function takes Google API Key and an audio file path as input, transcribes it using Whisper,
    summarizes it using Google Gemini and turns the summary back into speech.
    Args:
        api_key (str): The Google API Key.
        audio_path (str): The path to the audio file.
    Returns:
        tuple: A tuple containing the transcription, the summary and tuple containing the audio data.
    '''
    # Check if the API Key was provided, if not, return an error message
    if not api_key:
        return "API Key is required. Please provide a valid Gemini API Key", "N/A", None

    # Check if the audio was provided, if not, return an error message
    if audio_path is None:
        return "No audio provided.", "N/A", None

    try:
        # Transcribe the audio
        transcription = stt_pipeline(audio_path)["text"]
        
        # Create the model and Summarize
        # Configure SDK with user-provided secret key
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Define the prompt with the main instruction
        prompt = "Provide a concise summary of the following transcription, highlighting the main ideas and key points in bullet points. Keep the summary under 50 words."

        # Send the prompt and the audio transcription to the model
        response = model.generate_content(f"{prompt}\n\n{transcription}")

        # Save the summary from the response
        summary = response.text
 
        # Turn the summary back into speech
        audio_data = tts_pipeline(summary)
    except Exception as e:
        return "Error", f"Error: {str(e)}", None
    
    return transcription, summary, (audio_data["sampling_rate"], audio_data["audio"])

# Section 3 - Build the Gradio Interface
# Create a title and description for the app
with gr.Blocks(title="AI Audio Analyzer") as demo:
    gr.Markdown("# 🎙️ AI Audio Analyzer")
    gr.Markdown("Use the following tabs to play with Audio Analysis Models for Speech-to-Text, Text-to-Speech and Sentiment Analysis")

    # Tab 1 - Speech-to-Text and Sentiment Analyzer
    with gr.Tab("Speech-to-Text and Sentiment Analyzer"):
        gr.Markdown("# 🎙️ Speech-to-Text and Sentiment Analyzer")
        gr.Markdown("Record your voice or upload an audio file to see it transcribed and analyzed.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"], 
                    type="filepath", 
                    label="Input Audio"
                )
                submit_btn = gr.Button("Analyze Audio", variant="primary")
                
            with gr.Column():
                text_output = gr.Textbox(label="Output - Transcription", interactive=False)
                sentiment_output = gr.Label(label="Detected Sentiment")

        submit_btn.click(
            fn=process_audio, 
            inputs=audio_input, 
            outputs=[text_output, sentiment_output]
        )

    # Tab 2 - Text-to-Speech and Sentiment Analyzer
    with gr.Tab("Text-to-Speech and Sentiment Analyzer"):
        gr.Markdown("# 🎙️ Text-to-Speech and Sentiment Analyzer")
        gr.Markdown("Type some text to see it converted to speech and analyzed.")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Input Text", placeholder="Type or paste the text you want to analyze here.")
                submit_btn = gr.Button("Analyze and Speak", variant="primary")
                
            with gr.Column():
                audio_output = gr.Audio(label="Output - Generated Speech")
                sentiment_output = gr.Label(label="Detected Sentiment")

        submit_btn.click(
            fn=analyze_and_speak, 
            inputs=text_input, 
            outputs=[audio_output, sentiment_output]
        )
    
    # Tab 3 - AI Meeting Assistant
    with gr.Tab("AI Meeting Assistant"):
        gr.Markdown("# 📋 AI Meeting Assistant")
        gr.Markdown("Record a meeting transcript or upload a meeting audio file; the AI will transcribe it, summarize it, and read the summary back.")
        gr.Markdown("You will need to provide your Google Gemini API Key to use this feature. Get your API key at https://aistudio.google.com/app/apikey")

        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(label="Your Gemini API Key", placeholder="Enter your Gemini API Key here", type="password")
                audio_in = gr.Audio(label="Record Meeting Snippet", type="filepath")
                submit_btn = gr.Button("Process Audio", variant="primary")
        
            with gr.Column():
                text_out = gr.Textbox(label="Output - Full Transcription", lines=5)
                summary_out = gr.Textbox(label="Output - Key Summary")
                audio_out = gr.Audio(label="Summary Voice-Over")

        submit_btn.click(
            fn=process_meeting,
            inputs=[api_key_input, audio_in],
            outputs=[text_out, summary_out, audio_out]
        )

# Section 4 - Launch the application
if __name__ == "__main__":
    demo.launch()
