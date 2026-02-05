"""
Prompt Optimizer for SAM3 Auto-Labeling

This module provides optimized text prompts for better detection accuracy with SAM3.
Instead of using generic class names, it uses descriptive prompts that improve
detection precision and reduce false positives.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Optimizes text prompts for SAM3 detection.

    SAM3 performs better with descriptive prompts rather than single-word class names.
    This class provides:
    - Optimized prompt variants per class
    - Ensemble detection with multiple prompts
    - NMS to merge results from different prompts
    """

    # Optimized prompts for common classes
    # Each class has multiple prompt variants that capture different aspects
    OPTIMIZED_PROMPTS: Dict[str, List[str]] = {
        # Marine life
        "fish": ["a fish swimming", "fish in water", "tropical fish", "swimming fish"],
        "coral": ["coral reef", "sea coral", "coral formation"],
        "shark": ["a shark", "shark swimming", "large shark"],
        "turtle": ["sea turtle", "turtle swimming", "marine turtle"],
        "jellyfish": ["jellyfish floating", "a jellyfish", "translucent jellyfish"],
        "octopus": ["an octopus", "octopus with tentacles"],
        "starfish": ["a starfish", "sea star", "starfish on surface"],
        "crab": ["a crab", "crab with claws"],
        "dolphin": ["a dolphin", "dolphin swimming"],
        "whale": ["a whale", "large whale", "whale in ocean"],

        # Vehicles
        "car": ["a car", "automobile", "passenger car", "vehicle on road"],
        "truck": ["a truck", "cargo truck", "large truck"],
        "bus": ["a bus", "passenger bus", "city bus"],
        "motorcycle": ["a motorcycle", "motorbike", "two-wheeled vehicle"],
        "bicycle": ["a bicycle", "bike", "person on bicycle"],
        "boat": ["a boat", "small boat", "watercraft"],
        "airplane": ["an airplane", "aircraft", "plane in sky"],
        "train": ["a train", "railway train", "locomotive"],

        # People
        "person": ["a person", "human figure", "standing person", "walking person"],
        "face": ["a human face", "person's face", "facial features"],
        "hand": ["a human hand", "person's hand", "open hand"],
        "head": ["a human head", "person's head"],

        # Urban
        "building": ["a building", "structure", "tall building"],
        "traffic light": ["a traffic light", "traffic signal", "stoplight"],
        "stop sign": ["a stop sign", "red stop sign", "octagonal sign"],
        "street sign": ["a street sign", "road sign"],
        "bench": ["a bench", "park bench", "sitting bench"],
        "parking meter": ["a parking meter", "meter on street"],
        "fire hydrant": ["a fire hydrant", "red hydrant"],

        # Nature
        "bird": ["a bird", "flying bird", "perched bird"],
        "cat": ["a cat", "domestic cat", "feline"],
        "dog": ["a dog", "domestic dog", "canine"],
        "horse": ["a horse", "standing horse"],
        "sheep": ["a sheep", "woolly sheep"],
        "cow": ["a cow", "cattle", "bovine"],
        "elephant": ["an elephant", "large elephant"],
        "bear": ["a bear", "standing bear"],
        "zebra": ["a zebra", "striped zebra"],
        "giraffe": ["a giraffe", "tall giraffe"],
        "tree": ["a tree", "standing tree", "tree with branches"],
        "flower": ["a flower", "blooming flower", "colorful flower"],

        # Food
        "apple": ["an apple", "red apple", "fruit apple"],
        "banana": ["a banana", "yellow banana"],
        "orange": ["an orange", "citrus orange"],
        "pizza": ["a pizza", "pizza slice", "round pizza"],
        "sandwich": ["a sandwich", "food sandwich"],
        "cake": ["a cake", "decorated cake", "birthday cake"],
        "cup": ["a cup", "drinking cup", "coffee cup"],
        "bowl": ["a bowl", "food bowl"],
        "bottle": ["a bottle", "drinking bottle", "water bottle"],
        "knife": ["a knife", "kitchen knife"],
        "fork": ["a fork", "eating fork"],
        "spoon": ["a spoon", "eating spoon"],

        # Fashion
        "shirt": ["a shirt", "clothing shirt", "worn shirt"],
        "pants": ["pants", "trousers", "leg clothing"],
        "dress": ["a dress", "woman's dress", "clothing dress"],
        "shoe": ["a shoe", "footwear", "worn shoe"],
        "hat": ["a hat", "head covering", "worn hat"],
        "bag": ["a bag", "handbag", "carrying bag"],
        "tie": ["a tie", "necktie", "worn tie"],
        "watch": ["a watch", "wristwatch", "worn watch"],
        "glasses": ["glasses", "eyeglasses", "spectacles"],
        "backpack": ["a backpack", "carrying backpack", "worn backpack"],
    }

    # Negative prompts to help reduce false positives (for future use)
    NEGATIVE_PROMPTS: Dict[str, List[str]] = {
        "fish": ["rock", "seaweed", "coral texture", "sand"],
        "coral": ["rock", "sand", "debris"],
        "person": ["mannequin", "statue", "poster", "reflection"],
        "car": ["toy car", "car in picture", "building", "road texture"],
    }

    # Aspect ratio constraints per class (min, max)
    # Helps filter out detections with unrealistic proportions
    ASPECT_RATIO_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
        "fish": (0.3, 4.0),      # Fish can be elongated
        "person": (0.2, 1.5),    # People are usually taller than wide
        "car": (0.5, 3.0),       # Cars are wider than tall
        "truck": (0.4, 4.0),     # Trucks can be very long
        "bicycle": (0.5, 2.5),   # Bicycles are wider than tall
        "tree": (0.3, 3.0),      # Trees are usually taller than wide
    }

    def __init__(self, use_ensemble: bool = True, max_prompts: int = 3):
        """
        Initialize the PromptOptimizer.

        Args:
            use_ensemble: Whether to use multiple prompts per class
            max_prompts: Maximum number of prompts to use in ensemble
        """
        self.use_ensemble = use_ensemble
        self.max_prompts = max_prompts

    def get_prompts(self, class_name: str) -> List[str]:
        """
        Get optimized prompts for a class.

        Args:
            class_name: The class name to get prompts for

        Returns:
            List of optimized prompts (1 if ensemble disabled, up to max_prompts if enabled)
        """
        # Normalize class name
        normalized = class_name.lower().strip().replace("_", " ")

        # Get optimized prompts or fall back to class name
        prompts = self.OPTIMIZED_PROMPTS.get(normalized, [class_name])

        if not self.use_ensemble:
            return [prompts[0]]

        return prompts[:self.max_prompts]

    def get_primary_prompt(self, class_name: str) -> str:
        """
        Get the single best prompt for a class.

        Args:
            class_name: The class name

        Returns:
            The primary optimized prompt
        """
        prompts = self.get_prompts(class_name)
        return prompts[0] if prompts else class_name

    def get_aspect_ratio_constraint(self, class_name: str) -> Tuple[float, float]:
        """
        Get aspect ratio constraints for a class.

        Args:
            class_name: The class name

        Returns:
            Tuple of (min_ratio, max_ratio) where ratio = width / height
        """
        normalized = class_name.lower().strip().replace("_", " ")
        return self.ASPECT_RATIO_CONSTRAINTS.get(normalized, (0.1, 10.0))

    def merge_detections(
        self,
        all_detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Merge detections from multiple prompts using NMS.

        Args:
            all_detections: List of detections from different prompts
                           Each detection has: mask, score, bbox
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            Filtered list of unique detections
        """
        if len(all_detections) <= 1:
            return all_detections

        # Sort by score descending
        sorted_dets = sorted(all_detections, key=lambda x: x.get("score", 0), reverse=True)

        keep = []
        for det in sorted_dets:
            is_duplicate = False
            for kept in keep:
                iou = self._calculate_bbox_iou(det["bbox"], kept["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(det)

        return keep

    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            bbox1, bbox2: Bounding boxes in [x, y, w, h] format

        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to [x1, y1, x2, y2] format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Calculate intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def validate_aspect_ratio(
        self,
        bbox: List[float],
        class_name: str
    ) -> bool:
        """
        Validate if a detection's aspect ratio is reasonable for its class.

        Args:
            bbox: Bounding box in [x, y, w, h] format
            class_name: The detected class

        Returns:
            True if aspect ratio is valid, False otherwise
        """
        w, h = bbox[2], bbox[3]
        if h <= 0:
            return False

        ratio = w / h
        min_ratio, max_ratio = self.get_aspect_ratio_constraint(class_name)

        return min_ratio <= ratio <= max_ratio


# Global instance for easy access
_prompt_optimizer: Optional[PromptOptimizer] = None


def get_prompt_optimizer() -> PromptOptimizer:
    """Get or create the global PromptOptimizer instance."""
    global _prompt_optimizer
    if _prompt_optimizer is None:
        _prompt_optimizer = PromptOptimizer()
    return _prompt_optimizer
