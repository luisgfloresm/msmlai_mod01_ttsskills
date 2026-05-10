import gradio as gr
from transformers import pipeline
import torch

# 1. Initialize the Pipelines from Hugging Face
# Whisper is excellent for transcription; DistilBERT is perfect for fast sentiment analysis
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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

# 2. Build the Gradio Interface
with gr.Blocks(title="AI Audio Analyzer") as demo:
    gr.Markdown("# 🎙️ Speech-to-Text & Sentiment Analyzer")
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

# 3. Launch
if __name__ == "__main__":
    demo.launch(share=True)