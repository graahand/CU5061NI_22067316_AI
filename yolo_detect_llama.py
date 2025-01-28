import os
import time
import json
import matplotlib.pyplot as plt
from ollama import chat, ChatResponse
from google.cloud import texttospeech
from datetime import datetime

INPUT_DIR = "shared_frames"
PROCESSED_DIR = "processed_frames"
TTS_DIR = "tts_context"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

# Text-to-Speech synthesis function
def synthesize_speech(text, output_audio_path):
    """Synthesizes speech from the input text using Google Text-to-Speech API."""
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRALl
    )

    # Configure audio file format
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Perform text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write the response to the output audio file
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file: {output_audio_path}")

# Function to display slideshow of images
def display_slideshow(images):
    for image_path in images:
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Processed Frame")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

# Main processing function
def process_frames_with_llama():
    processed_files = set()

    while True:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]

        for file in files:
            if file in processed_files:
                continue

            file_path = os.path.join(INPUT_DIR, file)
            metadata_path = file_path.replace(".jpg", ".json")

            if not os.path.exists(metadata_path):
                print(f"Metadata not found for {file_path}, skipping.")
                continue

            with open(metadata_path, "r") as meta_file:
                metadata = json.load(meta_file)

            object_details = ", ".join(
                [f"{obj['class_name']} (confidence: {obj['confidence_score']:.2f})" for obj in metadata]
            )

            prompt = (
                f"A new frame has been captured and saved as: {file_path}. "
                f"Detected objects: {object_details}. "
                "Provide 40-words straight-forward context about the frame for knowing what is happening around the environment for visually impaired person."
            )

            print(f"Processing {file_path} with Llama...")
            response: ChatResponse = chat(model="llama3.2-vision", messages=[
                {'role': 'user', 'content': prompt}
            ])

            llama_response_text = response.message.content.strip()
            print("Llama Response:")
            print(llama_response_text)

            # Convert Llama's response to speech
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_output_name = f"{timestamp}_{os.path.splitext(file)[0]}.mp3"
            audio_output_path = os.path.join(TTS_DIR, audio_output_name)
            synthesize_speech(llama_response_text, audio_output_path)

            # Add processed file to the set
            processed_files.add(file)

            processed_path = os.path.join(PROCESSED_DIR, file)
            metadata_processed_path = os.path.join(PROCESSED_DIR, os.path.basename(metadata_path))
            os.rename(file_path, processed_path)
            os.rename(metadata_path, metadata_processed_path)

            # Display slideshow of processed image
            display_slideshow([processed_path])

            # Optionally play the audio file (requires an audio player installed)
            os.system(f"start {audio_output_path}")

            # Delete processed files
            os.remove(processed_path)
            os.remove(metadata_processed_path)
            print(f"Deleted processed image and metadata: {processed_path}, {metadata_processed_path}")

        time.sleep(1)

if __name__ == "__main__":
    process_frames_with_llama()
