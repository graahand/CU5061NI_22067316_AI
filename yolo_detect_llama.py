"""Artifiical Intelligence
    Bibek Poudel (22067316)"""

import os
import time
import json
import matplotlib.pyplot as plt
from ollama import chat, ChatResponse

INPUT_DIR = "shared_frames"
PROCESSED_DIR = "processed_frames"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def display_slideshow(images):
    for image_path in images:
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Processed Frame")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

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
            response: ChatResponse = chat(model="llama3.1", messages=[
                {'role': 'user', 'content': prompt}
            ])

            print("Llama Response:")
            print(response.message.content)

            processed_files.add(file)

            processed_path = os.path.join(PROCESSED_DIR, file)
            metadata_processed_path = os.path.join(PROCESSED_DIR, os.path.basename(metadata_path))
            os.rename(file_path, processed_path)
            os.rename(metadata_path, metadata_processed_path)

            display_slideshow([processed_path])

            os.remove(processed_path)
            os.remove(metadata_processed_path)
            print(f"Deleted processed image and metadata: {processed_path}, {metadata_processed_path}")

        time.sleep(1)

if __name__ == "__main__":
    process_frames_with_llama()
