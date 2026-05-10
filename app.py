import gradio as gr
from transformers import pipeline
import torch

# 1. Initialize the Pipelines from Hugging Face
# Whisper is excellent for transcription; DistilBERT is perfect for fast sentiment analysis
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 1. Initialize the Pipelines from Hugging Face
# TTS Model: Meta's MMS (Massively Multilingual Speech) - Fast and high quality
tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-eng")


def process_audio(audio_path):
    if audio_path is None:
        return "No audio detected", "N/A"
    
    # Step 1: Speech to Text
    # Whisper handles the audio file path directly
    transcription = stt_pipeline(audio_path)["text"]
    
    # Step 2: Sentiment Analysis
    sentiment_result = sentiment_pipeline(transcription)[0]
    label = sentiment_result['label']
    score = round(sentiment_result['score'], 4)
    
    sentiment_output = f"{label} (Confidence: {score})"
    
    return transcription, sentiment_output

def analyze_and_speak(text):
    if not text.strip():
        return None, "Please enter some text."

    # Part A: Generate Sentiment Analysis
    sentiment_result = sentiment_pipeline(text)[0]
    label = sentiment_result['label']
    score = sentiment_result['score']
    sentiment_output = f"Analysis: {label} ({score:.2%} confidence)"

    # Part B: Generate Audio (TTS)
    # The pipeline returns a dictionary with 'audio' (numpy array) and 'sampling_rate'
    audio_data = tts_pipeline(text)
    
    # Gradio gr.Audio component expects a tuple of (sample_rate, data)
    return (audio_data["sampling_rate"], audio_data["audio"]), sentiment_output

# 2. Build the Gradio Interface
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
                text_output = gr.Textbox(label="Transcription", interactive=False)
                sentiment_output = gr.Label(label="Detected Sentiment")

        # Connect the logic
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
                text_input = gr.Textbox(label="Input Text")
                submit_btn = gr.Button("Analyze and Speak", variant="primary")
                
            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech")
                sentiment_output = gr.Label(label="Detected Sentiment")

        # Connect the logic
        submit_btn.click(
            fn=analyze_and_speak, 
            inputs=text_input, 
            outputs=[audio_output, sentiment_output]
        )

# 3. Launch
if __name__ == "__main__":
    demo.launch(share=True)