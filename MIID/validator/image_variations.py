# MIID/validator/image_variations.py
#
# Phase 4: Image variation type definitions and random selection.
# Defines variation types with intensity bins for face image variations.

import random
from typing import List, Dict, Any, Optional


# =============================================================================
# Shared image requirement text (attached to every variation request)
# =============================================================================

# This text is appended to every VariationRequest so each variation the miner
# generates carries the same composition/resolution constraints.
IMAGE_VARIATION_REQUIREMENTS = (
    "All images are Professional passport-style portraits, 3:4 aspect ratio, "
    "head-and-shoulders composition from chest up. "
    "Recommended output resolution: 1015 x 1350 pixels."
)


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
        "description": "Change the background while keeping the subject unchanged",
        "intensities": {
            "light": {
                "label": "Light background change",
                "detail": "Light background change while preserving identity"
            },
            "medium": {
                "label": "Medium background change",
                "detail": "Medium background change while preserving identity"
            },
            "far": {
                "label": "Far background change",
                "detail": "Far background change while preserving identity"
            }
        }
    }
}

BACKGROUND_ENVIRONMENT_TYPES = {
    "indoor": {
        "description": "Indoor background change",
        "weight": 70,
        "intensities": {
            "light": {
                "label": "Minor indoor background adjustment",
                "detail": "Same indoor context; wall color shift, bokeh/blur, or subtle texture change"
            },
            "medium": {
                "label": "Different indoor setting",
                "detail": "Switch to another plausible indoor environment (e.g. office to cafe, bedroom to lobby); still fully indoors"
            },
            "far": {
                "label": "Dramatic indoor background change",
                "detail": "Move to a clearly different indoor setting with distinct interior design and scene depth (e.g., office, lobby, studio, gallery), while keeping the background fully indoors with no outdoor elements"
            }
        }
    },
    "outdoor": {
        "description": "Outdoor background change",
        "weight": 30,
        "intensities": {
            "light": {
                "label": "Minor outdoor background adjustment",
                "detail": "Same outdoor context with subtle changes (e.g. slight depth-of-field shift, mild scene texture variation, or small lighting-direction change)"
            },
            "medium": {
                "label": "Different outdoor setting",
                "detail": "Switch to a different plausible outdoor environment (e.g. street to park, plaza to waterfront) while keeping the subject unchanged"
            },
            "far": {
                "label": "Dramatic outdoor background change",
                "detail": "Move to a clearly different outdoor setting with distinct scene composition and depth (e.g., urban street, beach promenade, mountain overlook, garden path)"
            }
        }
    }
}

# S3 / request type names for the two background slots. Never emit a bare
# "background" or "background_edit" type — miners upload
# .../{background_in|background_out}_{timestamp}.png.tlock
BACKGROUND_INDOOR_TYPE = "background_in"
BACKGROUND_OUTDOOR_TYPE = "background_out"
BACKGROUND_SLOT_TYPES = (BACKGROUND_INDOOR_TYPE, BACKGROUND_OUTDOOR_TYPE)


def background_type_for_environment(environment_key: str) -> str:
    """Map indoor/outdoor to the miner-facing variation type used in S3 keys."""
    if environment_key == "outdoor":
        return BACKGROUND_OUTDOOR_TYPE
    return BACKGROUND_INDOOR_TYPE


def is_background_variation_type(var_type: str) -> bool:
    """True for indoor/outdoor background slots (and legacy background_edit)."""
    return var_type in BACKGROUND_SLOT_TYPES or var_type == "background_edit"


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


def select_random_background_environment() -> Dict[str, Any]:
    """Select indoor/outdoor background environment using weighted distribution."""
    environment_keys = list(BACKGROUND_ENVIRONMENT_TYPES.keys())
    weights = [BACKGROUND_ENVIRONMENT_TYPES[key]["weight"] for key in environment_keys]
    selected_key = random.choices(environment_keys, weights=weights, k=1)[0]
    selected_info = BACKGROUND_ENVIRONMENT_TYPES[selected_key]
    return {
        "type": selected_key,
        "description": selected_info["description"],
        "intensities": selected_info["intensities"],
    }


def get_background_variation_info(intensity: str) -> Dict[str, str]:
    """Get weighted-random indoor/outdoor background detail for an intensity."""
    environment = select_random_background_environment()
    intensity_info = environment["intensities"][intensity]
    return {
        "environment_type": environment["type"],
        "environment_description": environment["description"],
        "label": intensity_info["label"],
        "detail": intensity_info["detail"],
    }


def get_background_variation_info_for_environment(
    intensity: str, environment_key: str
) -> Dict[str, str]:
    """Background detail for a fixed environment (``indoor`` or ``outdoor``) and intensity."""
    selected_info = BACKGROUND_ENVIRONMENT_TYPES[environment_key]
    intensity_info = selected_info["intensities"][intensity]
    return {
        "environment_type": environment_key,
        "environment_description": selected_info["description"],
        "label": intensity_info["label"],
        "detail": intensity_info["detail"],
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
        if var_type == "background_edit":
            intensity_info = get_background_variation_info(intensity)
            description = f"{type_info['description']}. {intensity_info['environment_description']}"
            detail = intensity_info["detail"]
            emitted_type = background_type_for_environment(intensity_info["environment_type"])
        else:
            intensity_info = type_info["intensities"][intensity]
            description = type_info["description"]
            detail = intensity_info["detail"]
            emitted_type = var_type

        variations.append({
            "type": emitted_type,
            "intensity": intensity,
            "description": description,
            "detail": detail
        })

    return variations


def format_variation_requirements(variations: List[Dict[str, str]]) -> str:
    """Format variation requirements as text for query template.

    Creates a human-readable description of the requested variations
    to be appended to the query template sent to miners.
    
    If a background_in / background_out slot is included without an accessory
    already in the detail, automatically adds a random accessory.

    Args:
        variations: List of variation dicts from select_random_variations()

    Returns:
        Formatted string describing the variation requirements

    Example output:
        [IMAGE VARIATION REQUIREMENTS]
        For the face image provided, generate the following variations while preserving identity. All images are Professional passport-style portraits, 3:4 aspect ratio, head-and-shoulders composition from chest up.

        1. pose_edit (medium): ±30° rotation (clear head turn, profile partially visible)
        2. expression_edit (light): Neutral to slight smile, minor brow movement
        3. background_in (far): Move to a clearly different indoor setting with distinct interior design and scene depth (e.g., office, lobby, studio, gallery), while keeping the background fully indoors with no outdoor elements. Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject

        IMPORTANT: The subject's face must remain recognizable across all variations.
    """
    lines = [
        "",
        "[IMAGE VARIATION REQUIREMENTS]",
        f"For the face image provided, generate the following variations while preserving identity. {IMAGE_VARIATION_REQUIREMENTS}",
        ""
    ]

    # Only draw an accessory if the background variation detail doesn't already include one.
    # This avoids "double-randomizing" when background_edit was produced by get_variation_by_index()
    # or by our dedicated helpers.
    needs_accessory = any(
        is_background_variation_type(var.get("type", ""))
        and ("Additionally, include:" not in var.get("detail", ""))
        for var in variations
    )
    accessory = select_random_accessory() if needs_accessory else None

    for i, var in enumerate(variations, 1):
        detail = var['detail']
        # If this is a background slot and detail doesn't already include accessory, add it once
        if is_background_variation_type(var['type']) and accessory and "Additionally, include:" not in detail:
            detail = f"{detail}. Additionally, include: {accessory['detail']}"
        lines.append(
            f"{i}. {var['type']} ({var['intensity']}): {detail}"
        )

    lines.extend([
        "",
        "IMPORTANT: The subject's face must remain recognizable across all variations.",
        "Each variation should clearly address all specified types and intensity levels "
        "(combined variations must satisfy every component).",
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
    if var_type == "background_edit":
        intensity_info = get_background_variation_info(intensity)
        description = f"{type_info['description']}. {intensity_info['environment_description']}"
        detail = intensity_info["detail"]
        label = intensity_info["label"]
        emitted_type = background_type_for_environment(intensity_info["environment_type"])
    else:
        intensity_info = type_info["intensities"][intensity]
        description = type_info["description"]
        detail = intensity_info["detail"]
        label = intensity_info["label"]
        emitted_type = var_type

    return {
        "type": emitted_type,
        "intensity": intensity,
        "description": description,
        "label": label,
        "detail": detail
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
        if (
            var["type"] not in ALL_VARIATION_TYPES
            and var["type"] not in BACKGROUND_SLOT_TYPES
        ):
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
    """Get a background_in or background_out variation (from the indoor/outdoor draw) + accessory."""
    intensity = random.choice(ALL_INTENSITIES)

    type_info = IMAGE_VARIATION_TYPES["background_edit"]
    intensity_info = get_background_variation_info(intensity)

    accessory = select_random_accessory()
    description = (
        f"{type_info['description']}. {intensity_info['environment_description']}. "
        f"{accessory['description']}"
    )
    detail = f"{intensity_info['detail']}. Additionally, include: {accessory['detail']}"

    return {
        "type": background_type_for_environment(intensity_info["environment_type"]),
        "intensity": intensity,
        "description": description,
        "detail": detail,
    }


def get_random_indoor_background_variation() -> Dict[str, str]:
    """Get a ``background_in`` variation constrained to indoor settings + weighted accessory."""
    intensity = random.choice(ALL_INTENSITIES)
    type_info = IMAGE_VARIATION_TYPES["background_edit"]
    intensity_info = get_background_variation_info_for_environment(intensity, "indoor")

    accessory = select_random_accessory()
    description = (
        f"{type_info['description']}. {intensity_info['environment_description']}. "
        f"{accessory['description']}"
    )
    detail = f"{intensity_info['detail']}. Additionally, include: {accessory['detail']}"

    return {
        "type": BACKGROUND_INDOOR_TYPE,
        "intensity": intensity,
        "description": description,
        "detail": detail,
    }


def get_random_outdoor_background_variation() -> Dict[str, str]:
    """Get a ``background_out`` variation constrained to outdoor settings + weighted accessory."""
    intensity = random.choice(ALL_INTENSITIES)
    type_info = IMAGE_VARIATION_TYPES["background_edit"]
    intensity_info = get_background_variation_info_for_environment(intensity, "outdoor")

    accessory = select_random_accessory()
    description = (
        f"{type_info['description']}. {intensity_info['environment_description']}. "
        f"{accessory['description']}"
    )
    detail = f"{intensity_info['detail']}. Additionally, include: {accessory['detail']}"

    return {
        "type": BACKGROUND_OUTDOOR_TYPE,
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
    return get_random_variation_by_type(var_type)


def get_random_variation_by_type(var_type: str) -> Dict[str, str]:
    """Get one random-intensity variation for a specific type."""
    if var_type == "background_edit" or var_type in BACKGROUND_SLOT_TYPES:
        if var_type == BACKGROUND_INDOOR_TYPE:
            return get_random_indoor_background_variation()
        if var_type == BACKGROUND_OUTDOOR_TYPE:
            return get_random_outdoor_background_variation()
        return get_random_background_variation()
    intensity = random.choice(ALL_INTENSITIES)
    type_info = IMAGE_VARIATION_TYPES[var_type]
    intensity_info = type_info["intensities"][intensity]

    return {
        "type": var_type,
        "intensity": intensity,
        "description": type_info["description"],
        "detail": intensity_info["detail"],
    }


COMBINED_VARIATION_SEPARATOR = "+"


def get_random_combined_variation(var_types: List[str]) -> Dict[str, Any]:
    """Get a single image variation that combines multiple edit types.

    Each component gets its own random intensity. The returned ``type`` is the
    component keys joined with ``+`` (e.g. ``lighting_edit+expression_edit``),
    which miners must use as ``variation_type`` when uploading.

    Args:
        var_types: Two or more variation type keys from IMAGE_VARIATION_TYPES.

    Returns:
        Dict with type, intensity, description, detail, and components metadata.
    """
    if len(var_types) < 2:
        raise ValueError("Combined variation requires at least 2 types")

    components: List[Dict[str, str]] = []
    for var_type in var_types:
        if var_type not in IMAGE_VARIATION_TYPES:
            raise ValueError(f"Unknown variation type: {var_type}")
        intensity = random.choice(ALL_INTENSITIES)
        type_info = IMAGE_VARIATION_TYPES[var_type]
        intensity_info = type_info["intensities"][intensity]
        components.append({
            "type": var_type,
            "intensity": intensity,
            "description": type_info["description"],
            "detail": intensity_info["detail"],
        })

    combined_type = COMBINED_VARIATION_SEPARATOR.join(var_types)
    combined_intensity = COMBINED_VARIATION_SEPARATOR.join(c["intensity"] for c in components)
    combined_description = (
        "Combined variation — apply all of the following while preserving identity: "
        + "; ".join(
            f"{c['type']} ({c['intensity']}): {c['description']}" for c in components
        )
    )
    combined_detail = " AND ".join(
        f"{c['type']} ({c['intensity']}): {c['detail']}" for c in components
    )

    return {
        "type": combined_type,
        "intensity": combined_intensity,
        "description": combined_description,
        "detail": combined_detail,
        "components": components,
    }


def build_standard_challenge_variations() -> List[Dict[str, Any]]:
    """Build the standard 5-variation synthetic (FLUX-generated) challenge set.

    Order:
    1. background_in (indoor)
    2. background_out (outdoor)
    3. lighting_edit + expression_edit
    4. lighting_edit + pose_edit
    5. pose_edit + expression_edit

    NOTE: screen_replay is intentionally NOT part of this set anymore. It is
    no longer a synthetic/FLUX-generated variation requested every round —
    it is a REAL physical screen capture using the daily fixed seed image.
    Miners may submit as many of these real captures as they want, whenever
    ready (no daily cap) — the only rule is that every submission must be a
    genuinely new capture (never a duplicate of one already sent), and each
    submission bundles two photos of the same capture as basic proof it's
    real: (1) a face-dominant, centered, low-distortion close-up of the
    screen, and (2) a wider environment shot of the whole device/scene.
    Those two files are reviewed for cross-view consistency (same seed face
    on screen in both views, same device/bezel geometry, consistent
    lighting/glare direction, distinct hashes). See
    format_real_screen_replay_instructions() for the miner-facing task
    text (including that review checklist), and ScreenReplayUAV in
    MIID/protocol.py for the reported metadata.
    """
    return [
        get_random_indoor_background_variation(),
        get_random_outdoor_background_variation(),
        get_random_combined_variation(["lighting_edit", "expression_edit"]),
        get_random_combined_variation(["lighting_edit", "pose_edit"]),
        get_random_combined_variation(["pose_edit", "expression_edit"]),
    ]


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
    if var_type == "background_edit":
        intensity_info = get_background_variation_info(intensity)
        detail = intensity_info["detail"]
        description = f"{type_info['description']}. {intensity_info['environment_description']}"
    else:
        intensity_info = type_info["intensities"][intensity]
        detail = intensity_info["detail"]
        description = type_info["description"]

    # For background slots only: append a random accessory (weighted) to description and detail
    if var_type == "background_edit":
        accessory = select_random_accessory()
        description = f"{description}. {accessory['description']}"
        detail = f"{detail}. Additionally, include: {accessory['detail']}"
        var_type = background_type_for_environment(intensity_info["environment_type"])

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

# Capture-variety tracks: how the seed was prepared before the real screen
# capture. Device/camera used are reported separately (camera_used /
# device_photographed). Each submission still needs a face close-up
# (photo or video) + an environment still of the same physical capture.
#
# Six options: 3 photo + 3 video.
SCREEN_REPLAY_CAPTURE_VARIANTS: Dict[str, Dict[str, str]] = {
    "seed_unchanged": {
        "label": "Seed unchanged",
        "primary_media": "photo",
        "summary": "Display the seed as-is, then photograph it.",
    },
    "seed_smiling": {
        "label": "Seed smiling",
        "primary_media": "photo",
        "summary": "Edit seed to smile, display it, then photograph it.",
    },
    "seed_eyes_closed": {
        "label": "Seed eyes closed",
        "primary_media": "photo",
        "summary": "Edit seed to eyes-closed, display it, then photograph it.",
    },
    "seed_video_blinking": {
        "label": "Seed video — blink",
        "primary_media": "video",
        "summary": "Make a blink seed-video, play it, record a real screen video.",
    },
    "seed_video_smiling": {
        "label": "Seed video — smile",
        "primary_media": "video",
        "summary": "Make a smile seed-video, play it, record a real screen video.",
    },
    "seed_video_smile_and_blink": {
        "label": "Seed video — smile + blink",
        "primary_media": "video",
        "summary": "Make a smile-and-blink seed-video, play it, record a real screen video.",
    },
}

# Older capture_variant names → current keys (queued submissions / old docs).
SCREEN_REPLAY_CAPTURE_VARIANT_ALIASES: Dict[str, str] = {
    "device_camera": "seed_unchanged",
    "synthetic_eyes_closed": "seed_eyes_closed",
    "synthetic_smiling": "seed_smiling",
    "synthetic_video_blinking": "seed_video_blinking",
    "synthetic_video_smiling": "seed_video_smiling",
    "synthetic_video_smile_and_blink": "seed_video_smile_and_blink",
    "synthetic_video_expression": "seed_video_smiling",  # legacy combined track
}

# Video capture_variant keys (primary media is a screen-replay video).
SCREEN_REPLAY_VIDEO_VARIANTS = frozenset(
    key
    for key, info in SCREEN_REPLAY_CAPTURE_VARIANTS.items()
    if info.get("primary_media") == "video"
)


def normalize_capture_variant(capture_variant: Optional[str]) -> str:
    """Map omitted/legacy capture_variant names to a current key."""
    raw = (capture_variant or "").strip() or "seed_unchanged"
    return SCREEN_REPLAY_CAPTURE_VARIANT_ALIASES.get(raw, raw)


def is_screen_replay_video_variant(capture_variant: str) -> bool:
    """True when this capture_variant's primary media is a video."""
    return normalize_capture_variant(capture_variant) in SCREEN_REPLAY_VIDEO_VARIANTS

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


def validate_screen_replay_uav(uav: Any) -> bool:
    """Validate a miner's ScreenReplayUAV checklist for a real screen-replay capture.

    Mirrors validate_variation_request() but for the UAV metadata attached to
    the screen_replay S3Submission: checks that free-text fields are present,
    device_photographed is a recognized device, capture_variant is one of the
    six variety tracks (3 photo + 3 video), and every cue key defined in
    SCREEN_REPLAY_VISUAL_CUES was reported as a bool (miners always report all
    five cues, regardless of how many are actually visible).

    Accepts either a ScreenReplayUAV instance (typical, since
    S3Submission.screen_replay_uav is already parsed by pydantic) or an
    equivalent plain dict.

    Args:
        uav: ScreenReplayUAV instance or dict to validate

    Returns:
        True if the checklist is well-formed
    """
    if uav is None:
        return False

    def _get(field: str) -> Any:
        return uav.get(field) if isinstance(uav, dict) else getattr(uav, field, None)

    for field in ("seed_image", "date", "camera_used", "device_photographed"):
        value = _get(field)
        if not isinstance(value, str) or not value.strip():
            return False

    if _get("device_photographed") not in SCREEN_REPLAY_DEVICE_TYPES:
        return False

    # Older submissions may omit capture_variant or use a legacy name.
    capture_variant = normalize_capture_variant(_get("capture_variant"))
    if capture_variant not in SCREEN_REPLAY_CAPTURE_VARIANTS:
        return False

    for cue_key in SCREEN_REPLAY_VISUAL_CUES:
        if not isinstance(_get(cue_key), bool):
            return False

    return True


# =============================================================================
# Real Screen Replay Instructions (physical capture, NOT FLUX-generated)
# =============================================================================

REAL_SCREEN_REPLAY_REQUIREMENTS = (
    "The PHOTOGRAPH / VIDEO of the screen must be REAL (physical camera, no "
    "screenshots). You MAY edit the seed first (smile, eyes closed, or a short "
    "blink / smile / smile+blink video), then display it and capture with a "
    "real camera. Device and camera are reported separately. "
    "Each submission needs TWO files of the SAME physical capture: "
    "(1) FACE CLOSE-UP — photo for the 3 still variants, or video for the 3 "
    "video variants; face dominant and centered, minimal angular distortion "
    "on stills. "
    "(2) ENVIRONMENT SHOT — always a still of the whole screen/device in its "
    "surroundings (distortion OK). "
    "Both files are reviewed together for cross-view consistency (same seed "
    "face on screen, same device/bezel, consistent lighting/glare, distinct "
    "hashes) — see the review checklist in the task instructions."
)


def build_screen_replay_uav_template(
    seed_filename: Optional[str] = None,
    seed_pool: Optional[List[str]] = None,
    tomorrow_seed_filename: Optional[str] = None,
    seed_date: Optional[str] = None,
    tomorrow_seed_date: Optional[str] = None,
) -> str:
    """Return a fill-in-the-blank ScreenReplayUAV template that miners copy-paste.

    The template is intentionally minimal: every field the miner must supply
    appears on its own line with a short placeholder.  The seed_image and date
    fields are pre-filled when the information is available. This checklist
    describes ONE capture event — the face close-up + environment shot
    submitted for that capture share this same metadata block.

    Args:
        seed_filename: Filename of today's validator-provided IOTD.
        seed_pool: Sandbox mode only — list of filenames from the shared
            fixed_image/ pool the miner picks from themselves.
        tomorrow_seed_filename: Filename of tomorrow's IOTD, sent early.
        seed_date: UTC date (YYYY-MM-DD) for today's IOTD.
        tomorrow_seed_date: UTC date (YYYY-MM-DD) for tomorrow's IOTD.

    Returns:
        A multi-line string block the miner copies, fills in, and attaches to
        their S3Submission as screen_replay_uav.
    """
    import datetime as _dt
    today_utc = seed_date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    device_options = "/".join(SCREEN_REPLAY_DEVICE_TYPES)
    variant_options = "/".join(SCREEN_REPLAY_CAPTURE_VARIANTS.keys())

    if seed_filename and tomorrow_seed_filename:
        seed_value = seed_filename
        seed_comment = (
            f"TODAY ({today_utc}); or \"{tomorrow_seed_filename}\" "
            f"for TOMORROW ({tomorrow_seed_date or 'UTC+1'})"
        )
    elif seed_filename:
        seed_value = seed_filename
        seed_comment = "DO NOT CHANGE — use the provided seed"
    elif seed_pool:
        seed_value = "FILL_IN_seed_filename"
        seed_comment = f"put the filename you randomly picked, one of: {', '.join(seed_pool)}"
    else:
        seed_value = "FILL_IN_seed_filename"
        seed_comment = "put the filename of the seed image you used"

    lines = [
        "# ════════════════════════════════════════════════",
        "# SCREEN-REPLAY SUBMISSION TEMPLATE  (copy & fill)",
        "# ════════════════════════════════════════════════",
        f'seed_image:               "{seed_value}"      # {seed_comment}',
        f'date:                     "{today_utc}"          # UTC capture date YYYY-MM-DD',
        'camera_used:              "YOUR_CAMERA_OR_PHONE"  # e.g. "iPhone 15 Pro"',
        f'device_photographed:      "phone"               # one of: {device_options}',
        f'capture_variant:          "seed_unchanged"      # one of: {variant_options}',
        'primary_media:             "photo"               # photo or video',
        "#",
        "# Mark each cue true if clearly visible in your capture, false otherwise.",
        "# Report honestly — graders will verify. Real captures may show 0–5 cues.",
        "#",
        "moire_pixel_grid:              false  # interference pattern from screen subpixels",
        "screen_glare_hotspots:         false  # specular reflections on display surface",
        "perspective_keystone_distortion: false  # geometric distortion from off-angle shot",
        "gamma_contrast_shift:          false  # colour/brightness of display capture",
        "edge_crop_cues:                false  # screen borders, bezel, or cropping visible",
        "# ════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


def format_real_screen_replay_instructions(
    seed_filename: Optional[str] = None,
    seed_pool: Optional[List[str]] = None,
    tomorrow_seed_filename: Optional[str] = None,
    seed_date: Optional[str] = None,
    tomorrow_seed_date: Optional[str] = None,
) -> str:
    """Build the miner-facing instructions for the real screen-replay task.

    Explains the physical capture: display a seed (or a synthetically edited
    seed / seed-video) on a real device screen, capture it with a different
    physical camera as a face close-up (photo or video) plus an environment
    still, and submit both together as one screen_replay submission with a
    filled-out ScreenReplayUAV report (including capture_variant).

    Six capture_variant tracks (3 photo + 3 video) — see
    SCREEN_REPLAY_CAPTURE_VARIANTS. Device/camera are separate UAV fields.

    Modes, controlled by which image the seed comes from:
      - Validator-provided IOTD (seed_filename set): miners use today's
        image-of-the-day, and optionally tomorrow's (sent early so they can
        prepare captures before UTC midnight).
      - Sandbox / miner-chosen (seed_pool set instead): the validator isn't
        sending a seed image. Miners pick one from the shared fixed_image/
        pool that ships with the codebase.

    Miners may send as many of these submissions as they want — there is no
    daily cap — but every submission must be a genuinely new capture. Never
    resubmit the same photos (or the same capture) twice; duplicates are
    filtered out and penalised.

    Args:
        seed_filename: Filename of today's validator-provided IOTD.
        seed_pool: List of filenames in the shared fixed_image/ pool (sandbox).
        tomorrow_seed_filename: Filename of tomorrow's IOTD, sent early.
        seed_date: UTC date (YYYY-MM-DD) for today's IOTD.
        tomorrow_seed_date: UTC date (YYYY-MM-DD) for tomorrow's IOTD.

    Returns:
        Formatted instructions string to send to miners alongside the request.
    """
    device_list = ", ".join(SCREEN_REPLAY_DEVICE_TYPES)

    if seed_filename and tomorrow_seed_filename:
        today_label = f"{seed_filename}" + (f" ({seed_date} UTC)" if seed_date else "")
        tomorrow_label = (
            f"{tomorrow_seed_filename}"
            + (f" ({tomorrow_seed_date} UTC)" if tomorrow_seed_date else "")
        )
        seed_line = (
            "Images of the day (from the validator — same pair for every miner today):\n"
            f"  • TODAY:     {today_label}\n"
            f"  • TOMORROW:  {tomorrow_label}\n"
            "Photograph TODAY's seed for captures you submit now. Tomorrow's seed is\n"
            "included so you can start capturing overnight / in advance — set\n"
            "seed_image to whichever filename you actually displayed."
        )
        display_base = "Start from TODAY's seed (or tomorrow's if you are prepping ahead)"
    elif seed_filename:
        seed_line = f"Today's seed image (from the validator): {seed_filename}"
        display_base = "Start from the provided seed image"
    elif seed_pool:
        pool_list = "\n".join(f"    - {name}" for name in seed_pool)
        seed_line = (
            "SANDBOX MODE: the validator isn't sending a seed image right now.\n"
            "Instead, randomly pick ONE image yourself from the shared pool below\n"
            "(MIID/validator/fixed_image/ in this repo — ships with the codebase,\n"
            "no download needed) and use it as your seed for this capture:\n"
            f"{pool_list}"
        )
        display_base = "Start from your randomly-chosen pool image"
    else:
        seed_line = "No seed image is currently configured for this task."
        display_base = "Start from your seed image"

    variant_lines = []
    for i, (key, info) in enumerate(SCREEN_REPLAY_CAPTURE_VARIANTS.items(), 1):
        media = info["primary_media"]
        variant_lines.append(f"  {i}. {key}  [{media}] — {info['label']}")
        variant_lines.append(f"     {info['summary']}")

    template_block = build_screen_replay_uav_template(
        seed_filename,
        seed_pool,
        tomorrow_seed_filename=tomorrow_seed_filename,
        seed_date=seed_date,
        tomorrow_seed_date=tomorrow_seed_date,
    )

    lines = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        "║  REAL SCREEN-REPLAY CAPTURE  (separate from synthetics) ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
        seed_line,
        "",
        REAL_SCREEN_REPLAY_REQUIREMENTS,
        "",
        "Send as MANY real captures as you like — there's no daily limit.",
        "The only rule: NEVER submit a duplicate. Every submission must be a",
        "fresh, genuinely new capture — resubmitting the same media again",
        "will be detected and your score WILL be penalised.",
        "",
        "Capture variety — pick ONE capture_variant per submission:",
        *variant_lines,
        "",
        "Every variant still needs TWO files of the SAME physical capture:",
        "  • Primary (FACE CLOSE-UP): photo for seed_unchanged / seed_smiling /",
        "    seed_eyes_closed; video for the three seed_video_* variants.",
        "  • Secondary (ENVIRONMENT): always a still of the whole device/scene.",
        "",
        "How your two files will be reviewed (cross-view consistency):",
        "  Manual review and automated flagging check that both files look like",
        "  the SAME physical capture of the same screen at the same moment:",
        "  • Same identity / seed face on the screen in the close-up AND in a",
        "    crop of the screen from the environment shot",
        "  • Same device / bezel geometry (same phone/tablet/laptop/monitor/tv",
        "    shape and framing of the display)",
        "  • Roughly the same lighting and glare direction across both shots",
        "  • Distinct file hashes — primary and environment must be two",
        "    different files (uploading the same file twice is rejected)",
        "  Capture both shots back-to-back of the same setup so these checks",
        "  pass. Do not pair unrelated photos or paste a new seed into an old",
        "  capture shell.",
        "",
        "Quick steps:",
        f"  1. {display_base}. Optionally edit it (smile / eyes closed / blink",
        "     video / smile video / smile+blink video), then display on a real",
        f"     device ({device_list}).",
        "  2. Capture with a DIFFERENT physical camera (no screenshots):",
        "       (a) FACE CLOSE-UP — photo or video per variant above.",
        "       (b) ENVIRONMENT — still photo of whole screen + surroundings.",
        "  3. Upload as variation_type=\"screen_replay\": primary in",
        "     s3_key/image_hash/signature, environment in s3_key_angle2/",
        "     image_hash_angle2/signature_angle2.",
        "  4. Fill in the template below ONCE and attach as screen_replay_uav",
        "     — including capture_variant and seed_image filename.",
        "",
        template_block,
        "",
        "Submit to ANY validator whenever you have a new capture ready — not",
        "tied to this request. Send as many non-duplicate captures as you can.",
        "",
    ]
    return "\n".join(lines)


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
            if var_type == "background_edit":
                intensity_info = get_background_variation_info(intensity)
                description = f"{type_info['description']}. {intensity_info['environment_description']}"
                detail = intensity_info["detail"]
                emitted_type = background_type_for_environment(intensity_info["environment_type"])
            else:
                intensity_info = type_info["intensities"][intensity]
                description = type_info["description"]
                detail = intensity_info["detail"]
                emitted_type = var_type
            combinations.append({
                "type": emitted_type,
                "intensity": intensity,
                "description": description,
                "detail": detail
            })
    return combinations
