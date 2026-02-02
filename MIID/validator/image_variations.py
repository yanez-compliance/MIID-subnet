# MIID/validator/image_variations.py
#
# Phase 4: Image variation type definitions and random selection.
# Defines variation types with intensity bins for face image variations.

import random
from typing import List, Dict, Any


# =============================================================================
# Image Variation Type Definitions (YEVS-style)
# =============================================================================

IMAGE_VARIATION_TYPES = {
    "pose_edit": {
        "description": "Change head pose (yaw/pitch/roll) while keeping identity",
        "intensities": {
            "light": {
                "label": "Light pose change",
                "detail": "±15° rotation (slight head tilt or turn)"
            },
            "medium": {
                "label": "Medium pose change",
                "detail": "±30° rotation (clear head turn, profile partially visible)"
            },
            "far": {
                "label": "Far pose change",
                "detail": ">±45° rotation (near-profile view, significant angle change)"
            }
        }
    },
    "lighting_edit": {
        "description": "Modify illumination direction, intensity, or color temperature",
        "intensities": {
            "light": {
                "label": "Mild lighting adjustment",
                "detail": "Subtle brightness or contrast change, soft shadows"
            },
            "medium": {
                "label": "Directional lighting change",
                "detail": "Clear directional light source, noticeable shadows, moderate intensity shift"
            },
            "far": {
                "label": "Extreme lighting change",
                "detail": "Strong shadows, dramatic contrast, unusual color temperature (warm/cool cast)"
            }
        }
    },
    "expression_edit": {
        "description": "Change facial expression while preserving identity",
        "intensities": {
            "light": {
                "label": "Subtle expression change",
                "detail": "Neutral to slight smile, minor brow movement, relaxed to attentive"
            },
            "medium": {
                "label": "Clear expression change",
                "detail": "Neutral to smile, serious, or mildly surprised expression"
            },
            "far": {
                "label": "Strong expression change",
                "detail": "Laughing, surprised, concerned, or other pronounced expression"
            }
        }
    },
    "background_edit": {
        "description": "Change background environment while keeping subject unchanged",
        "intensities": {
            "light": {
                "label": "Minor background adjustment",
                "detail": "Color shift, blur adjustment, or texture change on similar background"
            },
            "medium": {
                "label": "Different background setting",
                "detail": "Change environment type (office to outdoor, solid color to gradient)"
            },
            "far": {
                "label": "Dramatic background change",
                "detail": "Unusual or contrasting environment, complex scene, dramatic setting"
            }
        }
    }
}

# All available variation type keys
ALL_VARIATION_TYPES = list(IMAGE_VARIATION_TYPES.keys())

# All available intensity levels
ALL_INTENSITIES = ["light", "medium", "far"]


def select_random_variations(
    min_variations: int = 2,
    max_variations: int = 4
) -> List[Dict[str, str]]:
    """Randomly select variation types with random intensities.

    Each challenge gets a random subset of variation types, each with
    a randomly assigned intensity level. This prevents miners from
    gaming the system with fixed responses.

    Args:
        min_variations: Minimum number of variation types to select (default: 2)
        max_variations: Maximum number of variation types to select (default: 4)

    Returns:
        List of dicts, each containing:
            - type: str - variation type key (e.g., "pose_edit")
            - intensity: str - intensity level (e.g., "medium")
            - description: str - type description
            - detail: str - intensity-specific detail

    Example:
        >>> variations = select_random_variations()
        >>> print(variations)
        [
            {"type": "pose_edit", "intensity": "medium", "description": "...", "detail": "..."},
            {"type": "expression_edit", "intensity": "light", "description": "...", "detail": "..."},
        ]
    """
    # Determine how many variations to request (2-4)
    num_variations = random.randint(min_variations, max_variations)

    # Randomly select which types to include
    selected_types = random.sample(ALL_VARIATION_TYPES, num_variations)

    # Assign random intensity to each selected type
    variations = []
    for var_type in selected_types:
        intensity = random.choice(ALL_INTENSITIES)
        type_info = IMAGE_VARIATION_TYPES[var_type]
        intensity_info = type_info["intensities"][intensity]

        variations.append({
            "type": var_type,
            "intensity": intensity,
            "description": type_info["description"],
            "detail": intensity_info["detail"]
        })

    return variations


def format_variation_requirements(variations: List[Dict[str, str]]) -> str:
    """Format variation requirements as text for query template.

    Creates a human-readable description of the requested variations
    to be appended to the query template sent to miners.

    Args:
        variations: List of variation dicts from select_random_variations()

    Returns:
        Formatted string describing the variation requirements

    Example output:
        [IMAGE VARIATION REQUIREMENTS]
        For the face image provided, generate the following variations while preserving identity:

        1. pose_edit (medium): ±30° rotation (clear head turn, profile partially visible)
        2. expression_edit (light): Neutral to slight smile, minor brow movement
        3. background_edit (far): Unusual or contrasting environment, complex scene

        IMPORTANT: The subject's face must remain recognizable across all variations.
    """
    lines = [
        "",
        "[IMAGE VARIATION REQUIREMENTS]",
        "For the face image provided, generate the following variations while preserving identity:",
        ""
    ]

    for i, var in enumerate(variations, 1):
        lines.append(
            f"{i}. {var['type']} ({var['intensity']}): {var['detail']}"
        )

    lines.extend([
        "",
        "IMPORTANT: The subject's face must remain recognizable across all variations.",
        "Each variation should clearly address the specified type and intensity level.",
        ""
    ])

    return "\n".join(lines)


def get_variation_type_info(var_type: str, intensity: str) -> Dict[str, Any]:
    """Get full information for a specific variation type and intensity.

    Args:
        var_type: Variation type key (e.g., "pose_edit")
        intensity: Intensity level (e.g., "medium")

    Returns:
        Dict with type and intensity information

    Raises:
        KeyError: If var_type or intensity is invalid
    """
    type_info = IMAGE_VARIATION_TYPES[var_type]
    intensity_info = type_info["intensities"][intensity]

    return {
        "type": var_type,
        "intensity": intensity,
        "description": type_info["description"],
        "label": intensity_info["label"],
        "detail": intensity_info["detail"]
    }


def validate_variation_request(variations: List[Dict[str, str]]) -> bool:
    """Validate that a variation request is well-formed.

    Args:
        variations: List of variation dicts to validate

    Returns:
        True if all variations are valid
    """
    if not variations or not isinstance(variations, list):
        return False

    for var in variations:
        if not isinstance(var, dict):
            return False
        if "type" not in var or "intensity" not in var:
            return False
        if var["type"] not in ALL_VARIATION_TYPES:
            return False
        if var["intensity"] not in ALL_INTENSITIES:
            return False

    return True


def get_total_variation_combinations() -> int:
    """Get the total number of variation type + intensity combinations.

    Returns:
        Total number of combinations (types × intensities)
    """
    return len(ALL_VARIATION_TYPES) * len(ALL_INTENSITIES)


def get_variation_by_index(index: int) -> Dict[str, str]:
    """Get a single variation type + intensity by index.

    Cycles through all variation types, then all intensities for each type.
    Order: pose_edit/light, pose_edit/medium, pose_edit/far,
           lighting_edit/light, lighting_edit/medium, ...

    Supports wrapping around when index exceeds total combinations.

    Args:
        index: The index of the variation to get (will wrap around)

    Returns:
        Dict with type, intensity, description, and detail
    """
    total_combinations = get_total_variation_combinations()
    actual_index = index % total_combinations

    # Calculate which type and intensity
    type_index = actual_index // len(ALL_INTENSITIES)
    intensity_index = actual_index % len(ALL_INTENSITIES)

    var_type = ALL_VARIATION_TYPES[type_index]
    intensity = ALL_INTENSITIES[intensity_index]

    type_info = IMAGE_VARIATION_TYPES[var_type]
    intensity_info = type_info["intensities"][intensity]

    return {
        "type": var_type,
        "intensity": intensity,
        "description": type_info["description"],
        "detail": intensity_info["detail"]
    }


def get_all_variation_combinations() -> List[Dict[str, str]]:
    """Get all possible variation type + intensity combinations in order.

    Useful for debugging or understanding the full cycle.

    Returns:
        List of all variation dicts in sequential order
    """
    combinations = []
    for var_type in ALL_VARIATION_TYPES:
        type_info = IMAGE_VARIATION_TYPES[var_type]
        for intensity in ALL_INTENSITIES:
            intensity_info = type_info["intensities"][intensity]
            combinations.append({
                "type": var_type,
                "intensity": intensity,
                "description": type_info["description"],
                "detail": intensity_info["detail"]
            })
    return combinations
