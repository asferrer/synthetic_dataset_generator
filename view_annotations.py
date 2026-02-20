#!/usr/bin/env python3
"""
Simple script to visualize COCO annotations with OpenCV.
Press any key to advance to the next image, ESC to quit.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import random


def load_coco(json_path):
    """Load COCO annotations file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_random_color(seed=None):
    """Generate a random color for visualization."""
    if seed is not None:
        random.seed(seed)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def draw_annotation(image, annotation, categories, show_masks=True, show_boxes=True):
    """Draw a single annotation on the image."""
    # Get category info
    cat_id = annotation['category_id']
    category = next((c for c in categories if c['id'] == cat_id), None)
    if not category:
        return image

    class_name = category['name']
    color = get_random_color(cat_id)

    # Draw segmentation mask
    if show_masks and 'segmentation' in annotation and annotation['segmentation']:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for seg in annotation['segmentation']:
            if len(seg) < 6:  # Need at least 3 points
                continue
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Create colored overlay
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        # Blend with original image
        alpha = 0.4
        image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)

    # Draw bounding box
    if show_boxes and 'bbox' in annotation:
        x, y, w, h = [int(v) for v in annotation['bbox']]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        label = f"{class_name}"
        if '_score' in annotation:
            label += f" {annotation['_score']:.2f}"

        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - label_h - baseline - 5), (x + label_w, y), color, -1)
        cv2.putText(image, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def visualize_coco(coco_json_path, images_dirs=None, show_masks=True, show_boxes=True, shuffle=False):
    """Visualize COCO annotations."""
    coco_json_path = Path(coco_json_path)

    # Load COCO data
    print(f"Loading COCO data from {coco_json_path}")
    coco_data = load_coco(coco_json_path)

    # Determine images directories
    if images_dirs is None or len(images_dirs) == 0:
        # Try to find images in common locations
        possible_dirs = [
            coco_json_path.parent / "images",
            coco_json_path.parent.parent / "images",
            coco_json_path.parent,
        ]
        for possible_dir in possible_dirs:
            if possible_dir.exists() and any(possible_dir.glob("*.jpg")):
                images_dirs = [possible_dir]
                break

        if images_dirs is None:
            raise ValueError("Could not find images directory. Please specify with --images-dir")
    else:
        images_dirs = [Path(d) for d in images_dirs]

    print(f"Searching images in {len(images_dirs)} directories:")
    for img_dir in images_dirs:
        print(f"  - {img_dir}")

    # Get data
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])

    print(f"Found {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    print(f"Categories: {', '.join([c['name'] for c in categories])}")

    # Group annotations by image
    anns_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Shuffle if requested
    if shuffle:
        random.shuffle(images)

    # Visualize each image
    for img_info in images:
        img_id = img_info['id']
        file_name = img_info['file_name']

        # Search for image in all provided directories
        img_path = None
        for img_dir in images_dirs:
            candidate_path = img_dir / file_name
            if candidate_path.exists():
                img_path = candidate_path
                break

        if img_path is None:
            print(f"Warning: Image not found in any directory: {file_name}")
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Failed to load image: {img_path}")
            continue

        # Get annotations for this image
        img_anns = anns_by_image.get(img_id, [])

        # Draw annotations
        for ann in img_anns:
            image = draw_annotation(image, ann, categories, show_masks, show_boxes)

        # Add info text
        info_text = f"{file_name} - {len(img_anns)} annotations"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Resize if too large
        max_height = 1000
        h, w = image.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, max_height))

        # Show image
        cv2.imshow('COCO Annotations (Press any key for next, ESC to quit)', image)
        key = cv2.waitKey(0)

        # ESC key to quit
        if key == 27:
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations')
    parser.add_argument('coco_json', type=str, help='Path to COCO JSON file')
    parser.add_argument('--images-dir', type=str, action='append', dest='images_dirs',
                        help='Directory containing images (can be specified multiple times, auto-detected if not specified)')
    parser.add_argument('--no-masks', action='store_true',
                        help='Do not show segmentation masks')
    parser.add_argument('--no-boxes', action='store_true',
                        help='Do not show bounding boxes')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle images randomly')

    args = parser.parse_args()

    visualize_coco(
        args.coco_json,
        args.images_dirs,
        show_masks=not args.no_masks,
        show_boxes=not args.no_boxes,
        shuffle=args.shuffle
    )


if __name__ == '__main__':
    main()
