import os
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import chromadb

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
ROOT_IMAGE_FOLDER = "./flowers_images"  # Path to the root folder containing category folders
OUTPUT_FOLDER = "./clustered_flowers" # Base path for sorted results

# Vector Database Configuration
VECTOR_DB_PATH = "./flower_vector_db"
COLLECTION_NAME = "flower_embeddings"

# OPTION 1: Set to an integer if you know the exact number of flower categories per folder
# OPTION 2: Set to None to let the algorithm automatically cluster based on similarity
NUM_CLUSTERS = None  

# Distance threshold for auto-clustering
DISTANCE_THRESHOLD = 0.08 

MODEL_NAME = "facebook/dinov2-base"

# -------------------------------------------------------------------
# 1. Initialize Local Vector Database (ChromaDB)
# -------------------------------------------------------------------
print("Initializing local Vector Database...")
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# -------------------------------------------------------------------
# 2. Discover Root Folder Hierarchy
# -------------------------------------------------------------------
valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
root_abs_path = os.path.abspath(ROOT_IMAGE_FOLDER)

if not os.path.exists(root_abs_path):
    raise FileNotFoundError(f"Root directory not found: {root_abs_path}")

# Collect all (Category, Subfolder) folder tasks dynamically
folder_tasks = []

for category in os.listdir(root_abs_path):
    category_path = os.path.join(root_abs_path, category)
    if not os.path.isdir(category_path):
        continue

    for sub_folder in os.listdir(category_path):
        sub_folder_path = os.path.join(category_path, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue

        # Collect valid images in this leaf directory
        image_paths = [
            os.path.abspath(os.path.join(sub_folder_path, f))
            for f in os.listdir(sub_folder_path)
            if f.lower().endswith(valid_extensions)
        ]

        if image_paths:
            folder_tasks.append({
                "category": category,
                "sub_folder": sub_folder,
                "folder_abs_path": sub_folder_path,
                "image_paths": image_paths
            })

if not folder_tasks:
    raise ValueError(f"No valid image subfolders found under: {ROOT_IMAGE_FOLDER}")

print(f"Found {len(folder_tasks)} subfolder(s) across categories to process.")

# -------------------------------------------------------------------
# 3. Identify Missing Embeddings Across All Folders
# -------------------------------------------------------------------
paths_to_process = []

for task in folder_tasks:
    f_path = task["folder_abs_path"]
    db_records = collection.get(where={"folder_path": f_path})
    existing_ids = set(db_records["ids"]) if db_records and db_records["ids"] else set()
    
    missing_in_task = [p for p in task["image_paths"] if p not in existing_ids]
    paths_to_process.extend(missing_in_task)

# -------------------------------------------------------------------
# 4. Generate Embeddings (Single Model Pass for Efficiency)
# -------------------------------------------------------------------
if not paths_to_process:
    print("\n✅ All embeddings across all subfolders are already cached in Vector DB!")
    print("Skipping model loading and feature extraction.\n")
else:
    print(f"\nFound {len(paths_to_process)} new image(s) needing feature extraction.")
    print("Loading DINOv2 feature extractor model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with torch.no_grad():
        for img_path in tqdm(paths_to_process, desc="Extracting embeddings"):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # Extract [CLS] token vector
                feature_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                
                folder_dir = os.path.dirname(img_path)
                collection.add(
                    ids=[img_path],
                    embeddings=[feature_vector.tolist()],
                    metadatas=[{
                        "file_name": os.path.basename(img_path),
                        "path": img_path,
                        "folder_path": folder_dir
                    }]
                )
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# -------------------------------------------------------------------
# 5 & 6. Perform Clustering & Preserve Hierarchy in Output
# -------------------------------------------------------------------
print("\nStarting clustering and folder organization...")

for task in folder_tasks:
    category = task["category"]
    sub_folder = task["sub_folder"]
    f_path = task["folder_abs_path"]

    print(f"\nProcessing: [{category}] -> [{sub_folder}]")

    # Fetch stored embeddings for this specific folder
    db_contents = collection.get(
        where={"folder_path": f_path},
        include=["embeddings", "metadatas"]
    )

    # FIXED CODE
    if db_contents.get("embeddings") is None or len(db_contents["embeddings"]) == 0:
        print(f"⚠️ No embeddings found for {f_path}. Skipping.")
        continue

    embeddings = np.array(db_contents["embeddings"])
    valid_paths = [meta["path"] for meta in db_contents["metadatas"]]
    num_images = len(embeddings)

    # Edge case: Handle single image folders directly without clustering
    if num_images == 1:
        cluster_labels = np.array([0])
    else:
        normalized_embeddings = normalize(embeddings)

        if NUM_CLUSTERS is not None:
            clustering_model = AgglomerativeClustering(
                n_clusters=min(NUM_CLUSTERS, num_images), 
                metric="cosine", 
                linkage="average"
            )
        else:
            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD,
                metric="cosine",
                linkage="average"
            )

        cluster_labels = clustering_model.fit_predict(normalized_embeddings)

    n_found_clusters = len(set(cluster_labels))
    print(f"Identified {n_found_clusters} cluster(s) from {num_images} image(s).")

    # Target Directory: OUTPUT_FOLDER / Category_Name / Subfolder_Name / flower_type_XX
    target_base_dir = os.path.join(OUTPUT_FOLDER, category, sub_folder)

    for img_path, label in zip(valid_paths, cluster_labels):
        cluster_dir = os.path.join(target_base_dir, f"flower_type_{label + 1:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        file_name = os.path.basename(img_path)
        destination = os.path.join(cluster_dir, file_name)
        shutil.copy(img_path, destination)

print(f"\n Done! Clustered images saved under '{OUTPUT_FOLDER}' mirroring original structure.")