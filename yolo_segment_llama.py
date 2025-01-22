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
        plt.title("Processed Segmentation Frame")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def format_segmentation_details(metadata):
    if 'detections' in metadata:
        segments = metadata['detections']
    else:
        segments = metadata
    
    if not segments:
        return "The scene appears to be empty with no detected objects."
        
    object_groups = {}
    for seg in segments:
        class_name = seg.get('class_name', 'unknown')
        confidence = seg.get('confidence_score', 0)
        area = seg.get('area', 0)
        
        if class_name not in object_groups:
            object_groups[class_name] = []
        object_groups[class_name].append((confidence, area))
    
    descriptions = []
    for class_name, instances in object_groups.items():
        count = len(instances)
        avg_confidence = sum(conf for conf, _ in instances) / count
        total_area = sum(area for _, area in instances)
        
        if count == 1:
            size_desc = "small" if total_area < 2000 else "medium" if total_area < 5000 else "large"
            descriptions.append(f"one {size_desc} {class_name} with {avg_confidence:.1%} confidence")
        else:
            descriptions.append(f"{count} {class_name}s with average confidence of {avg_confidence:.1%}")
    
    return " and ".join(descriptions)

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
                continue
                
            try:
                with open(metadata_path, "r") as meta_file:
                    metadata = json.load(meta_file)
                
                scene_description = format_segmentation_details(metadata)
                
                prompt = (
                    f"You are describing a scene to a visually impaired person. "
                    f"The scene contains: {scene_description} "
                    "Based on this information, provide a natural, conversational 40-word description "
                    "of what the person would encounter in this space, focusing on the layout "
                    "and spatial relationships between objects."
                )

                response: ChatResponse = chat(model="llama3.1", messages=[
                    {'role': 'user', 'content': prompt}
                ])

                processed_files.add(file)
                processed_path = os.path.join(PROCESSED_DIR, file)
                metadata_processed_path = os.path.join(PROCESSED_DIR, os.path.basename(metadata_path))
                
                os.rename(file_path, processed_path)
                os.rename(metadata_path, metadata_processed_path)

                display_slideshow([processed_path])
                os.remove(processed_path)
                os.remove(metadata_processed_path)

            except json.JSONDecodeError:
                continue
            except Exception:
                continue

        time.sleep(1)

if __name__ == "__main__":
    try:
        process_frames_with_llama()
    except KeyboardInterrupt:
        print("\Shutting down...")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
