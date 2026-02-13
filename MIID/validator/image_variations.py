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

# Accessory types with weighted selection
ACCESSORY_TYPES = {
    "religious_head_covering": {
        "description": "Add religious head covering",
        "detail": "Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject",
        "weight": 50
    },
    "brim_hat": {
        "description": "Add brim hat (not baseball)",
        "detail": "Brim hat such as fedora, wide-brim hat, sun hat, or similar (not baseball cap)",
        "weight": 20
    },
    "knit_winter_hat": {
        "description": "Add knit or winter hat",
        "detail": "Knit hat, beanie, or winter hat",
        "weight": 20
    },
    "bandana": {
        "description": "Add bandana",
        "detail": "Bandana worn on head",
        "weight": 5
    },
    "baseball_cap": {
        "description": "Add baseball cap",
        "detail": "Baseball cap or similar sports cap",
        "weight": 5
    }
}

# All available variation type keys (excluding background which is always included)
OPTIONAL_VARIATION_TYPES = ["pose_edit", "lighting_edit", "expression_edit"]

# All available intensity levels
ALL_INTENSITIES = ["light", "medium", "far"]


def select_random_accessory() -> Dict[str, str]:
    """Select a random accessory based on weighted distribution.
    
    Weights:
    - 50% religious head coverings
    - 20% Brim hats (not baseball)
    - 20% Knit/winter hats
    - 5% Bandanas
    - 5% Baseball caps
    
    Returns:
        Dict containing:
            - type: str - accessory type key
            - description: str - accessory description
            - detail: str - specific accessory detail
    """
    # Create weighted list
    accessory_keys = list(ACCESSORY_TYPES.keys())
    weights = [ACCESSORY_TYPES[key]["weight"] for key in accessory_keys]
    
    # Select based on weights
    selected_key = random.choices(accessory_keys, weights=weights, k=1)[0]
    accessory_info = ACCESSORY_TYPES[selected_key]
    
    return {
        "type": selected_key,
        "description": accessory_info["description"],
        "detail": accessory_info["detail"]
    }


def select_random_variations(
    min_variations: int = 2,
    max_variations: int = 4
) -> List[Dict[str, str]]:
    """Select variation types with new fixed structure.

    Each challenge now includes:
    1. Background change (always included, random intensity)
    2. Accessory (always included, weighted random selection)
    3. One additional variation from: lighting, pose, or expression (random intensity)

    Args:
        min_variations: Ignored - kept for backward compatibility
        max_variations: Ignored - kept for backward compatibility

    Returns:
        List of dicts, each containing:
            - type: str - variation type key (e.g., "pose_edit", "accessory")
            - intensity: str - intensity level (e.g., "medium") or None for accessory
            - description: str - type description
            - detail: str - intensity-specific or accessory-specific detail

    Example:
        >>> variations = select_random_variations()
        >>> print(variations)
        [
            {"type": "background_edit", "intensity": "medium", "description": "...", "detail": "..."},
            {"type": "religious_head_covering", "intensity": None, "description": "...", "detail": "..."},
            {"type": "lighting_edit", "intensity": "light", "description": "...", "detail": "..."},
        ]
    """
    variations = []
    
    # 1. Always include background change with random intensity
    bg_intensity = random.choice(ALL_INTENSITIES)
    bg_info = IMAGE_VARIATION_TYPES["background_edit"]
    bg_intensity_info = bg_info["intensities"][bg_intensity]
    
    variations.append({
        "type": "background_edit",
        "intensity": bg_intensity,
        "description": bg_info["description"],
        "detail": bg_intensity_info["detail"]
    })
    
    # 2. Always include accessory with weighted selection
    accessory = select_random_accessory()
    variations.append({
        "type": accessory["type"],
        "intensity": "",  # Empty string for accessories (Pydantic requires string, not None)
        "description": accessory["description"],
        "detail": accessory["detail"]
    })
    
    # 3. Select one random variation from lighting, pose, or expression
    additional_type = random.choice(OPTIONAL_VARIATION_TYPES)
    additional_intensity = random.choice(ALL_INTENSITIES)
    additional_info = IMAGE_VARIATION_TYPES[additional_type]
    additional_intensity_info = additional_info["intensities"][additional_intensity]
    
    variations.append({
        "type": additional_type,
        "intensity": additional_intensity,
        "description": additional_info["description"],
        "detail": additional_intensity_info["detail"]
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
        Professional passport-style portrait, 3:4 aspect ratio, head and shoulders composition from chest up.
        
        For the face image provided, generate the following variations while preserving identity:

        1. background_edit (medium): Change environment type (office to outdoor, solid color to gradient)
        2. accessory (religious_head_covering): Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject
        3. lighting_edit (light): Subtle brightness or contrast change, soft shadows

        IMPORTANT: The subject's face must remain recognizable across all variations.
    """
    lines = [
        "",
        "[IMAGE VARIATION REQUIREMENTS]",
        "Professional passport-style portrait, 3:4 aspect ratio, head and shoulders composition from chest up.",
        "",
        "Generate one single image with all of the following variations applied simultaneously to the face image, while preserving identity:",
        ""
    ]

    for i, var in enumerate(variations, 1):
        # Format differently for accessories (which have empty intensity)
        if var.get('intensity') == "":
            lines.append(
                f"{i}. accessory ({var['type']}): {var['detail']}"
            )
        else:
            lines.append(
                f"{i}. {var['type']} ({var['intensity']}): {var['detail']}"
            )

    lines.extend([
        "",
        "Generate one image with all variations applied together, not separate images for each variation.",
        "The subject's face must remain recognizable (identity preserved) with all modifications applied simultaneously.",
        "Each variation should be clearly visible in the single combined output image.",
        "Maintain professional passport-style composition (3:4 ratio, chest up, face centered).",
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
        if "type" not in var:
            return False
        
        # Check if it's an accessory type
        if var["type"] in ACCESSORY_TYPES:
            # Accessories should have empty string intensity
            if var.get("intensity") != "":
                return False
        else:
            # Regular variations must have valid type and intensity
            if var["type"] not in IMAGE_VARIATION_TYPES:
                return False
            if "intensity" not in var or var["intensity"] not in ALL_INTENSITIES:
                return False

    return True


def get_total_variation_combinations() -> int:
    """Get the total number of variation combinations in the cycle.
    
    With the new sequential cycling:
    - Background: Random (not part of cycle)
    - Accessory: Random (not part of cycle)
    - Additional: 3 types × 3 intensities = 9 variations
    - Each variation is done 2 times
    
    Total cycle length: 9 × 2 = 18

    Returns:
        Total number of combinations in cycle (18)
    """
    # 9 additional variations (3 types × 3 intensities), each done twice
    num_additional_variations = len(OPTIONAL_VARIATION_TYPES) * len(ALL_INTENSITIES)  # 9
    repetitions_per_variation = 2
    
    return num_additional_variations * repetitions_per_variation  # 18


def get_variation_by_index(index: int) -> List[Dict[str, str]]:
    """Get a specific variation combination by index for sequential cycling.
    
    Sequential cycling logic:
    - Cycles through 9 additional variations (pose/lighting/expression × light/medium/far)
    - Each variation is done 2 times before moving to the next
    - Background: Random intensity (different each time)
    - Accessory: Random weighted selection (different each time)
    - Total cycle length: 18 (9 variations × 2 repetitions)
    
    Example sequence:
    - Index 0-1: pose(light) with random bg/accessory
    - Index 2-3: pose(medium) with random bg/accessory
    - Index 4-5: pose(far) with random bg/accessory
    - Index 6-7: lighting(light) with random bg/accessory
    - ...
    - Index 18: wraps back to index 0
    
    Args:
        index: The index of the variation combination to get (will wrap around)

    Returns:
        List of 3 variation dicts (background + accessory + additional)
    """
    total_combinations = get_total_variation_combinations()  # 18
    actual_index = index % total_combinations
    
    # Determine which of the 9 additional variations to use
    # Each variation appears twice, so divide by 2
    variation_index = actual_index // 2
    
    # Calculate which type and intensity for the additional variation
    additional_type_index = variation_index // len(ALL_INTENSITIES)
    additional_intensity_index = variation_index % len(ALL_INTENSITIES)
    
    additional_type = OPTIONAL_VARIATION_TYPES[additional_type_index]
    additional_intensity = ALL_INTENSITIES[additional_intensity_index]
    
    # Get RANDOM background variation (different each time)
    bg_intensity = random.choice(ALL_INTENSITIES)
    bg_info = IMAGE_VARIATION_TYPES["background_edit"]
    bg_intensity_info = bg_info["intensities"][bg_intensity]
    background_var = {
        "type": "background_edit",
        "intensity": bg_intensity,
        "description": bg_info["description"],
        "detail": bg_intensity_info["detail"]
    }
    
    # Get RANDOM accessory (weighted, different each time)
    accessory = select_random_accessory()
    accessory_var = {
        "type": accessory["type"],
        "intensity": "",  # Empty string for accessories (Pydantic requires string, not None)
        "description": accessory["description"],
        "detail": accessory["detail"]
    }
    
    # Get deterministic additional variation (based on cycle position)
    additional_info = IMAGE_VARIATION_TYPES[additional_type]
    additional_intensity_info = additional_info["intensities"][additional_intensity]
    additional_var = {
        "type": additional_type,
        "intensity": additional_intensity,
        "description": additional_info["description"],
        "detail": additional_intensity_info["detail"]
    }
    
    # Return as list of 3 variations
    return [background_var, accessory_var, additional_var]


def get_all_variation_combinations() -> List[List[Dict[str, str]]]:
    """Get all variation combinations in the sequential cycle.

    NOTE: Since background and accessory are now random, this function
    returns only the deterministic additional variations. Each is shown
    twice to represent the 18-position cycle.
    
    The actual background and accessory will be random when called.

    Returns:
        List of all 18 cycle positions (9 variations × 2 repetitions)
    """
    all_combinations = []
    
    # Iterate through all additional variation types and intensities
    for additional_type in OPTIONAL_VARIATION_TYPES:
        additional_info = IMAGE_VARIATION_TYPES[additional_type]
        
        for additional_intensity in ALL_INTENSITIES:
            additional_intensity_info = additional_info["intensities"][additional_intensity]
            
            # Each variation appears twice in the cycle
            for repetition in range(2):
                # Create a sample combination (bg and accessory are random in actual use)
                combination = [
                    {
                        "type": "background_edit",
                        "intensity": "random",  # Placeholder - will be random in actual use
                        "description": IMAGE_VARIATION_TYPES["background_edit"]["description"],
                        "detail": "(Random - light/medium/far)"
                    },
                    {
                        "type": "random_accessory",  # Placeholder - will be weighted random in actual use
                        "intensity": "",  # Empty string for accessories (Pydantic requires string, not None)
                        "description": "Random accessory (weighted)",
                        "detail": "(50% religious, 20% brim, 20% knit, 5% bandana, 5% baseball)"
                    },
                    {
                        "type": additional_type,
                        "intensity": additional_intensity,
                        "description": additional_info["description"],
                        "detail": additional_intensity_info["detail"]
                    }
                ]
                
                all_combinations.append(combination)
    
    return all_combinations
