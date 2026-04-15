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
        "description": "Change the indoor background only while keeping the subject unchanged (no outdoor scenes, skyline, or open sky)",
        "intensities": {
            "light": {
                "label": "Minor indoor background adjustment",
                "detail": "Same indoor context; wall color shift, bokeh/blur, or subtle texture change"
            },
            "medium": {
                "label": "Different indoor setting",
                "detail": "Switch to another plausible indoor environment (e.g. office to café, bedroom to lobby); still fully indoors"
            },
            "far": {
                "label": "Dramatic indoor background change",
                "detail": "Move to a clearly different indoor setting with distinct interior design and scene depth (e.g., office, lobby, studio, gallery), while keeping the background fully indoors with no outdoor elements"
            }
        }
    }
}

# Accessory types with weighted selection
HEADPHONES_STYLE_OPTIONS = [
    "over-ear headphones",
    "on-ear headset",
    "studio headset",
]
HEADPHONES_COLOR_OPTIONS = ["black", "white", "dark gray", "navy"]
HEADPHONES_MATERIAL_OPTIONS = [
    "matte plastic",
    "composite polymer",
    "metal-reinforced plastic",
]
HEADPHONES_TEXTURE_OPTIONS = [
    "matte finish",
    "soft-touch finish",
    "smooth satin finish",
]
HEADPHONES_SIZE_FIT_OPTIONS = ["compact fit", "standard fit", "slightly oversized fit"]
HEADPHONES_EXTRA_DETAIL_OPTIONS = [
    "padded headband",
    "minimal branding",
    "simple ear-cup design",
    "fold-flat profile",
]


def select_random_headphones_detail() -> str:
    """Create a randomized headphones detail string from attribute options."""
    style = random.choice(HEADPHONES_STYLE_OPTIONS)
    color = random.choice(HEADPHONES_COLOR_OPTIONS)
    material = random.choice(HEADPHONES_MATERIAL_OPTIONS)
    texture = random.choice(HEADPHONES_TEXTURE_OPTIONS)
    size_fit = random.choice(HEADPHONES_SIZE_FIT_OPTIONS)
    extra_detail = random.choice(HEADPHONES_EXTRA_DETAIL_OPTIONS)
    return (
        f"{style} in {color} with a {material} frame, {texture}, {size_fit}, "
        f"and {extra_detail}"
    )


ACCESSORY_TYPES = {
    "religious_head_covering": {
        "description": "Add religious head covering",
        "detail": "Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject",
        "weight": 65
    },
    "brim_hat": {
        "description": "Add brim hat (not baseball)",
        "detail": "Brim hat such as fedora, wide-brim hat, sun hat, or similar (not baseball cap)",
        "weight": 10
    },
    "knit_winter_hat": {
        "description": "Add knit or winter hat",
        "detail": "Knit hat, beanie, or winter hat",
        "weight": 10
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
    },
    "headphones": {
        "description": "Add headphones",
        "detail": "Headphones appropriate to the subject; pick a realistic modern style",
        "weight": 5,
    }
}

# All available variation type keys
ALL_VARIATION_TYPES = list(IMAGE_VARIATION_TYPES.keys())

# All available intensity levels
ALL_INTENSITIES = ["light", "medium", "far"]


def select_random_accessory() -> Dict[str, str]:
    """Select a random accessory based on weighted distribution.
    
    Weights:
    - 65% religious head coverings
    - 10% Brim hats (not baseball)
    - 10% Knit/winter hats
    - 5% Bandanas
    - 5% Baseball caps
    - 5% Headphones
    
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
    
    detail = accessory_info["detail"]
    if selected_key == "headphones":
        detail = select_random_headphones_detail()

    return {
        "type": selected_key,
        "description": accessory_info["description"],
        "detail": detail,
    }


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
    
    If background_edit is included, automatically adds a random accessory
    to the background variation prompt.

    Args:
        variations: List of variation dicts from select_random_variations()

    Returns:
        Formatted string describing the variation requirements

    Example output:
        [IMAGE VARIATION REQUIREMENTS]
        For the face image provided, generate the following variations while preserving identity. All images are Professional passport-style portraits, 3:4 aspect ratio, head-and-shoulders composition from chest up.

        1. pose_edit (medium): ±30° rotation (clear head turn, profile partially visible)
        2. expression_edit (light): Neutral to slight smile, minor brow movement
        3. background_edit (far): Move to a clearly different indoor setting with distinct interior design and scene depth (e.g., office, lobby, studio, gallery), while keeping the background fully indoors with no outdoor elements. Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject

        IMPORTANT: The subject's face must remain recognizable across all variations.
    """
    lines = [
        "",
        "[IMAGE VARIATION REQUIREMENTS]",
        "For the face image provided, generate the following variations while preserving identity. All images are Professional passport-style portraits, 3:4 aspect ratio, head-and-shoulders composition from chest up.",
        ""
    ]

    # Only draw an accessory if the background variation detail doesn't already include one.
    # This avoids "double-randomizing" when background_edit was produced by get_variation_by_index()
    # or by our dedicated helpers.
    needs_accessory = any(
        (var.get("type") == "background_edit")
        and ("Additionally, include:" not in var.get("detail", ""))
        for var in variations
    )
    accessory = select_random_accessory() if needs_accessory else None

    for i, var in enumerate(variations, 1):
        detail = var['detail']
        # If this is background_edit and detail doesn't already include accessory (e.g. from get_variation_by_index), add it once
        if var['type'] == 'background_edit' and accessory and "Additionally, include:" not in detail:
            detail = f"{detail}. Additionally, include: {accessory['detail']}"
        lines.append(
            f"{i}. {var['type']} ({var['intensity']}): {detail}"
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


_NON_BACKGROUND_VARIATION_TYPES: List[str] = [t for t in ALL_VARIATION_TYPES if t != "background_edit"]


def get_total_non_background_variation_combinations() -> int:
    """Total combinations for all non-background types (pose/lighting/expression) × intensities."""
    return len(_NON_BACKGROUND_VARIATION_TYPES) * len(ALL_INTENSITIES)


def get_random_background_variation() -> Dict[str, str]:
    """Get a `background_edit` variation with random intensity + weighted accessory."""
    intensity = random.choice(ALL_INTENSITIES)

    type_info = IMAGE_VARIATION_TYPES["background_edit"]
    intensity_info = type_info["intensities"][intensity]

    accessory = select_random_accessory()
    description = f"{type_info['description']}. {accessory['description']}"
    detail = f"{intensity_info['detail']}. Additionally, include: {accessory['detail']}"

    return {
        "type": "background_edit",
        "intensity": intensity,
        "description": description,
        "detail": detail,
    }


def get_non_background_variation_by_index(index: int) -> Dict[str, str]:
    """Get a single non-background variation (pose/lighting/expression) by index (wraps)."""
    total_combinations = get_total_non_background_variation_combinations()
    actual_index = index % total_combinations

    type_index = actual_index // len(ALL_INTENSITIES)
    intensity_index = actual_index % len(ALL_INTENSITIES)

    var_type = _NON_BACKGROUND_VARIATION_TYPES[type_index]
    intensity = ALL_INTENSITIES[intensity_index]

    type_info = IMAGE_VARIATION_TYPES[var_type]
    intensity_info = type_info["intensities"][intensity]

    return {
        "type": var_type,
        "intensity": intensity,
        "description": type_info["description"],
        "detail": intensity_info["detail"],
    }


def get_random_non_background_variation() -> Dict[str, str]:
    """Get one random non-background variation (pose/lighting/expression)."""
    var_type = random.choice(_NON_BACKGROUND_VARIATION_TYPES)
    intensity = random.choice(ALL_INTENSITIES)

    type_info = IMAGE_VARIATION_TYPES[var_type]
    intensity_info = type_info["intensities"][intensity]

    return {
        "type": var_type,
        "intensity": intensity,
        "description": type_info["description"],
        "detail": intensity_info["detail"],
    }


def get_variation_by_index(index: int) -> Dict[str, str]:
    """Get a single variation type + intensity by index.

    Cycles through variation types in order: background, pose, lighting, expression;
    for each type, cycles through intensities: light, medium, far.
    Order: background_edit/light, background_edit/medium, background_edit/far,
           pose_edit/light, pose_edit/medium, pose_edit/far,
           lighting_edit/light, ..., expression_edit/far.

    When the variation is background_edit, a random accessory (weighted selection)
    is included in both description and detail (e.g. description "Change background... Add religious head covering").

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

    detail = intensity_info["detail"]
    description = type_info["description"]

    # For background_edit only: append a random accessory (weighted) to description and detail
    if var_type == "background_edit":
        accessory = select_random_accessory()
        description = f"{description}. {accessory['description']}"
        detail = f"{detail}. Additionally, include: {accessory['detail']}"

    return {
        "type": var_type,
        "intensity": intensity,
        "description": description,
        "detail": detail
    }


# =============================================================================
# Screen Replay Variation (Cycle 2 — not part of sequential image cycle)
# =============================================================================

# Device types — uniform random selection, no weights
SCREEN_REPLAY_DEVICE_TYPES = ["phone", "tablet", "laptop", "monitor", "tv"]

# Visual cues that must visibly appear in a screen-replay image (≥2 required)
SCREEN_REPLAY_VISUAL_CUES: Dict[str, str] = {
    "moire_pixel_grid": (
        "Moiré / pixel grid — interference pattern from screen subpixels captured by camera"
    ),
    "screen_glare_hotspots": (
        "Screen glare hotspots — specular reflections on the display surface"
    ),
    "perspective_keystone_distortion": (
        "Perspective / keystone distortion — geometric distortion from off-angle capture"
    ),
    "gamma_contrast_shift": (
        "Gamma / contrast shift — colour/brightness characteristics of display capture"
    ),
    "edge_crop_cues": (
        "Edge / crop cues — screen borders, bezel reflections, or cropping consistent with "
        "display capture"
    ),
}


def select_screen_replay_variation() -> Dict[str, str]:
    """Select a screen_replay variation with 2 random device types and 2 random visual cues.

    Device types are chosen uniformly at random (no weights) from
    SCREEN_REPLAY_DEVICE_TYPES.  Two visual cues are drawn without
    replacement from SCREEN_REPLAY_VISUAL_CUES.

    The returned dict is compatible with VariationRequest (type, intensity,
    description, detail) so no protocol changes are required.

    Returns:
        Dict with keys: type, intensity, description, detail
    """
    selected_device = random.choice(SCREEN_REPLAY_DEVICE_TYPES)
    selected_cue_keys = random.sample(list(SCREEN_REPLAY_VISUAL_CUES.keys()), 2)
    selected_cues = [SCREEN_REPLAY_VISUAL_CUES[k] for k in selected_cue_keys]

    cues_str = "; ".join(selected_cues)

    description = (
        f"Screen replay capture — simulate a photo of a face displayed on "
        f"a {selected_device} screen photographed by a physical camera"
    )
    detail = (
        f"Generate a realistic screen-replay image: a face shown on a {selected_device} "
        f"screen and photographed with a camera. "
        f"Must visibly exhibit at least 2 of these cues: {cues_str}. "
        f"The face must be the dominant object (large enough for reliable face detection) "
        f"and must remain matchable to the seed identity (high similarity score)."
    )

    return {
        "type": "screen_replay",
        "intensity": "standard",
        "description": description,
        "detail": detail,
        # Extra metadata carried in the plain dict (not sent over the wire via VariationRequest)
        "device_type": selected_device,
        "visual_cue_keys": selected_cue_keys,
    }


# =============================================================================


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
