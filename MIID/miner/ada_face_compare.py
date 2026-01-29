#!/usr/bin/env python3
"""
Face identity comparison using AdaFace.

This module is used by the miner to validate that generated image variations
preserve the identity of the original face. It loads a pretrained AdaFace model
(ir_50 on MS1MV2), extracts normalized face embeddings via MTCNN alignment,
and compares embeddings using cosine similarity. A variation is considered
identity-preserving if similarity is at or above the threshold (default 0.7).

  - validate_single_variation(base_image, variation_image, model=None, min_similarity=0.7, device='cpu') -> bool
  - compare_faces(original_image_path, variation_image_paths, model=None, device='cpu') -> dict

Prerequisites:
  1. Clone the AdaFace repository:
     git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace

  2. Install dependencies (if not already installed):
     pip install opencv-python torch pillow numpy

  3. Download a pretrained AdaFace model from:
     https://github.com/mk-minchul/AdaFace#pretrained-models
     Recommended: ir_50 model trained on MS1MV2

  4. Place the model file at:
     MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
"""

import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image

# Check for required dependencies
import cv2

# Add AdaFace to path (relative to this file)
_this_dir = os.path.dirname(os.path.abspath(__file__))
ADA_FACE_PATH = os.path.join(_this_dir, "AdaFace")
if os.path.isdir(ADA_FACE_PATH):
    sys.path.insert(0, ADA_FACE_PATH)

try:
    from face_alignment import align
    from face_alignment import mtcnn
    from inference import load_pretrained_model, to_input
    
    # Override the hardcoded CUDA device in align.py
    # Determine device based on CUDA availability
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    align.mtcnn_model = mtcnn.MTCNN(device=device, crop_size=(112, 112))
    
except ImportError as e:
    raise ImportError(
        f"AdaFace modules could not be imported: {e}. "
        f"Ensure AdaFace is cloned at {ADA_FACE_PATH}. See module docstring for setup."
    ) from e


def load_adaface_model(architecture='ir_50', model_path=None, device='cpu'):
    """
    Load AdaFace model for face recognition.
    
    Args:
        architecture: Model architecture ('ir_50', 'ir_101', etc.)
        model_path: Optional path to pretrained model. If None, uses default from inference.py
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded AdaFace model
    """
    import inference
    
    # Set default model path to absolute path if not provided
    if model_path is None:
        model_path = os.path.join(ADA_FACE_PATH, "pretrained", "adaface_ir50_ms1mv2.ckpt")
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(model_path):
        model_path = os.path.join(ADA_FACE_PATH, model_path)
    
    # Update the model path in inference module
    inference.adaface_models[architecture] = model_path
    
    # Verify file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please download it from: https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing"
        )
    
    print(f"Loading AdaFace model: {architecture}")
    print(f"Model path: {model_path}")
    
    # Temporarily override torch.load to use CPU map_location and weights_only=False
    import torch
    original_load = torch.load
    
    def cpu_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cpu'
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False  # Required for PyTorch 2.6+ with checkpoint files
        return original_load(*args, **kwargs)
    
    # Monkey patch torch.load for this call
    torch.load = cpu_load
    
    try:
        model = load_pretrained_model(architecture)
    finally:
        # Restore original torch.load
        torch.load = original_load
    
    model.eval()
    
    # Move model to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    
    print("✓ Model loaded successfully")
    return model


def extract_face_embedding(model, image_path, device='cpu'):
    """
    Extract face embedding from an image using AdaFace.
    
    Args:
        model: AdaFace model
        image_path: Path to image file
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        Face embedding tensor (normalized) or None if face detection fails
    """
    try:
        # Align face using MTCNN
        aligned_rgb_img = align.get_aligned_face(image_path)
        
        if aligned_rgb_img is None:
            print(f"⚠ Warning: Face detection failed for {image_path}")
            return None
        
        # Convert to BGR tensor input (AdaFace expects BGR)
        bgr_input = to_input(aligned_rgb_img)
        
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            bgr_input = bgr_input.cuda()
            model = model.cuda()
        
        # Extract feature embedding
        with torch.no_grad():
            feature, norm = model(bgr_input)
        
        # Normalize feature
        feature = feature / torch.norm(feature, 2, dim=1, keepdim=True)
        
        return feature.cpu()
    
    except Exception as e:
        print(f"✗ Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
    
    Returns:
        Cosine similarity score (0-1, where 1 is identical)
    """
    if embedding1 is None or embedding2 is None:
        return None
    
    # Compute cosine similarity
    similarity = torch.mm(embedding1, embedding2.t()).item()
    
    # Clamp to [0, 1] range (though cosine similarity is typically [-1, 1])
    # For face recognition, we usually expect positive similarities
    similarity = max(0.0, similarity)
    
    return similarity


def _path_from_image(img):
    """Return (path, cleanup_path). If img is a path str, return (img, None). If PIL Image, write to temp file and return (path, path)."""
    if isinstance(img, str) and os.path.isfile(img):
        return img, None
    if isinstance(img, Image.Image):
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(f.name, format="PNG")
        return f.name, f.name
    raise TypeError("base_image and variation_image must be a file path (str) or PIL Image")


def validate_single_variation(base_image, variation_image, model=None, min_similarity=0.7, device='cpu'):
    """
    Validate that a single variation image preserves the identity of the base face.
    Used by image_generator.validate_variation and by the miner to filter variations.

    Args:
        base_image: Original face image as file path (str) or PIL Image
        variation_image: Variation image as file path (str) or PIL Image
        model: AdaFace model (if None, will be loaded)
        min_similarity: Minimum cosine similarity to consider identity preserved (default 0.7)
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        True if similarity >= min_similarity, False otherwise (or if embedding extraction fails)
    """
    base_path, _ = _path_from_image(base_image)
    var_path, _ = _path_from_image(variation_image)
    results = compare_faces(base_path, [var_path], model=model, device=device)
    similarity = results.get(var_path)
    if similarity is None:
        return False
    return similarity >= min_similarity


def compare_faces(original_image_path, variation_image_paths, model=None, device='cpu'):
    """
    Compare original face with variation faces.
    
    Args:
        original_image_path: Path to original face image
        variation_image_paths: List of paths to variation images
        model: AdaFace model (if None, will be loaded)
        device: Device to run inference on
    
    Returns:
        Dictionary mapping variation paths to similarity scores
    """
    # Load model if not provided
    if model is None:
        model = load_adaface_model(device=device)
        # Initialize MTCNN if not already done
        if not hasattr(align, 'mtcnn_model') or align.mtcnn_model is None:
            mtcnn_device = 'cuda:0' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
            align.mtcnn_model = align.mtcnn.MTCNN(device=mtcnn_device, crop_size=(112, 112))
    
    print(f"\nExtracting embedding from original image: {original_image_path}")
    original_embedding = extract_face_embedding(model, original_image_path, device=device)
    
    if original_embedding is None:
        print("✗ Failed to extract embedding from original image. Cannot proceed.")
        return {}
    
    print("✓ Original embedding extracted")
    
    # Compare with each variation
    results = {}
    
    for var_path in variation_image_paths:
        print(f"\nProcessing variation: {var_path}")
        var_embedding = extract_face_embedding(model, var_path, device=device)
        
        if var_embedding is None:
            print(f"⚠ Skipping {var_path} - face detection failed")
            results[var_path] = None
            continue
        
        similarity = compute_cosine_similarity(original_embedding, var_embedding)
        results[var_path] = similarity
        
        print(f"✓ Similarity score: {similarity:.4f}")
    
    return results
