import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# File paths
val_dir = "hw2_data/clip_zeroshot/val"
id2label_path = "hw2_data/clip_zeroshot/id2label.json"

# Load id-to-label mapping
with open(id2label_path, "r") as f:
    id_to_label = json.load(f)

# Prepare text prompts for each class
class_labels = list(id_to_label.values())
prompts = [f"A photo of {label}." for label in class_labels]

# Encode text prompts
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Helper function to get class id from filename
def get_class_id(filename):
    return int(filename.split("_")[0])

# Variables for tracking results
all_preds = []
all_labels = []
success_cases = []
fail_cases = []

# Process each image in the validation directory
print("Processing images...")
for filename in tqdm(os.listdir(val_dir)):
    if not filename.endswith(".png"):
        continue
    
    # Get the true label for the image
    true_class_id = get_class_id(filename)
    true_label = id_to_label[str(true_class_id)]
    
    # Load and preprocess the image
    image_path = os.path.join(val_dir, filename)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Encode the image
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity between image and text features
    similarity = (image_features @ text_features.T).squeeze(0)
    predicted_index = similarity.argmax().item()
    predicted_label = class_labels[predicted_index]
    
    # Track predictions and ground truth
    all_preds.append(predicted_label)
    all_labels.append(true_label)
    
    # Save successful and failed cases
    if predicted_label == true_label:
        success_cases.append((filename, true_label, predicted_label))
    else:
        fail_cases.append((filename, true_label, predicted_label))

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")

# Report 5 successful and 5 failed cases
print("\n5 Successful Cases:")
for case in success_cases[:5]:
    filename, true_label, predicted_label = case
    print(f"File: {filename}, True: {true_label}, Predicted: {predicted_label}")

print("\n5 Failed Cases:")
for case in fail_cases[:5]:
    filename, true_label, predicted_label = case
    print(f"File: {filename}, True: {true_label}, Predicted: {predicted_label}")
