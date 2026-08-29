import os
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.preprocessing import normalize

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
INPUT_CLUSTERED_FOLDER = "./clustered_flowers"      # Output folder from previous script
OUTPUT_REP_FOLDER = "./representative_flowers"     # Target folder for representative images
MODEL_NAME = "facebook/dinov2-base"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# -------------------------------------------------------------------
# 1. Discover All Cluster Folders
# -------------------------------------------------------------------
input_abs_path = os.path.abspath(INPUT_CLUSTERED_FOLDER)

if not os.path.exists(input_abs_path):
    raise FileNotFoundError(f"Input directory not found: {input_abs_path}")

cluster_tasks = []

# Walk through the clustered directory tree
for root, dirs, files in os.walk(input_abs_path):
    image_paths = [
        os.path.join(root, f) 
        for f in files 
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    
    # Process folders that contain clustered images
    if image_paths:
        rel_path = os.path.relpath(root, input_abs_path)
        path_parts = rel_path.split(os.sep)
        
        # Expecting path structure: Category / Subfolder / flower_type_XX
        cluster_tasks.append({
            "full_path": root,
            "rel_path_parts": path_parts,
            "image_paths": image_paths
        })

if not cluster_tasks:
    raise ValueError(f"No cluster image folders found in: {INPUT_CLUSTERED_FOLDER}")

print(f"Found {len(cluster_tasks)} cluster folder(s) to process.")

# Check if any cluster needs neural net inference (> 1 image)
multi_image_clusters = [t for t in cluster_tasks if len(t["image_paths"]) > 1]

# -------------------------------------------------------------------
# 2. Lazy Load Model (Only if multi-image clusters exist)
# -------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None

if multi_image_clusters:
    print(f"\nLoading {MODEL_NAME} on {device.upper()} for representative selection...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

# -------------------------------------------------------------------
# 3. Find Representative Image per Cluster & Save to Output
# -------------------------------------------------------------------
print("\nProcessing clusters...")

for task in tqdm(cluster_tasks, desc="Finding Representatives"):
    img_paths = task["image_paths"]
    path_parts = task["rel_path_parts"]
    
    # Path components
    cluster_folder_name = path_parts[-1]             # e.g., 'flower_type_01'
    category_subfolder_rel = path_parts[:-1]         # e.g., ['Chandramallika', 'Single']
    
    # Create target directory mirroring input: OUTPUT_REP_FOLDER / Category / Subfolder
    target_dir = os.path.join(OUTPUT_REP_FOLDER, *category_subfolder_rel)
    os.makedirs(target_dir, exist_ok=True)
    
    best_img_path = None

    # Case 1: Single image cluster (No model run needed)
    if len(img_paths) == 1:
        best_img_path = img_paths[0]
        
    # Case 2: Multi-image cluster -> Compute Centroid Nearest Neighbor
    else:
        embeddings = []
        valid_img_paths = []

        with torch.no_grad():
            for p in img_paths:
                try:
                    image = Image.open(p).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    
                    # Extract [CLS] vector
                    vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                    embeddings.append(vec)
                    valid_img_paths.append(p)
                except Exception as e:
                    print(f"\nError processing {p}: {e}")

        if not embeddings:
            continue

        embeddings = np.array(embeddings)
        
        # Normalize vectors for Cosine Similarity
        norm_embeddings = normalize(embeddings)
        
        # Compute cluster centroid vector
        centroid = np.mean(norm_embeddings, axis=0, keepdims=True)
        norm_centroid = normalize(centroid)
        
        # Compute cosine similarity of each image to the centroid
        cosine_similarities = np.dot(norm_embeddings, norm_centroid.T).squeeze()
        
        # Pick image with highest similarity to centroid
        best_idx = int(np.argmax(cosine_similarities))
        best_img_path = valid_img_paths[best_idx]

    # Copy the selected representative image to destination
    if best_img_path:
        orig_filename = os.path.basename(best_img_path)
        
        # Name format: flower_type_XX_<original_name>
        dest_filename = f"{cluster_folder_name}_{orig_filename}"
        dest_path = os.path.join(target_dir, dest_filename)
        
        shutil.copy(best_img_path, dest_path)

print(f"\n Done! Representative images saved to '{OUTPUT_REP_FOLDER}' maintaining input folder structure.")