# Real Screen-Replay Photo Submission

This folder is where you submit a **real, physical capture** — photograph or
video of a seed displayed on a real screen (not a fully AI-faked room/device).
This is completely **optional** and is not required every round. There is **no
daily limit** — submit as many different real captures as you want. The only
hard rule is: **never submit a duplicate**.

> **Sandbox mode (current):** the validator isn't sending you a seed image
> right now. Instead, pick any ONE image yourself, at random, from
> `MIID/validator/fixed_image/` — a small pool that ships with this repo.

## Capture variants (pick one per submission)

Even with **one screen and one camera**, you can diversify submissions.
**Five options total: 2 photo + 3 video.**

| `#` | `capture_variant` | What you do | Primary media |
|---|---|---|---|
| 1 | `device_camera` | Display the seed as-is. Vary screen and/or camera when you have more than one. | Photo |
| 2 | `synthetic_eyes_closed` | Synthesize an **eyes-closed** version of the seed (keep identity), display *that*, then real-capture. | Photo |
| 3 | `synthetic_video_blinking` | Synthesize a short seed **video of blinking**, play it on screen, record a real screen-replay **video**, plus environment still. | Video |
| 4 | `synthetic_video_smiling` | Synthesize a short seed **video of smiling**, play it on screen, record a real screen-replay **video**, plus environment still. | Video |
| 5 | `synthetic_video_smile_and_blink` | Synthesize a short seed **video of smiling while blinking**, play it on screen, record a real screen-replay **video**, plus environment still. | Video |

Every variant still needs an **environment still photo** of the whole
screen/device in its surroundings (distortion OK) so the capture can be
corroborated as physical.

## TL;DR

1. Pick ONE image at random from `MIID/validator/fixed_image/`.
2. Choose a `capture_variant` (table above). For variants 2–5, synthesize the
   edited seed / seed-video first, then display it on a real screen.
3. Capture with a **different** physical camera (no screenshots):
   - **Primary — FACE CLOSE-UP:** face dominant + centered, minimal angular
     distortion (photo for 1–2, video for 3–5).
   - **Secondary — ENVIRONMENT:** wider still of the whole device/scene.
4. Drop both files into `inbox/` — exactly 2 images for variants 1–2, or
   **1 video + 1 image** for variants 3–5. iPhone HEIC is auto-converted.
5. Run:
   ```bash
   python MIID/miner/real_image_miner_guide/submit_real_photo.py
   ```
6. Answer prompts (variant, roles, seed, camera, device, cues). Your miner
   picks it up on the next validator query.

## How it works under the hood

- `submit_real_photo.py` stages the media and writes `screen_replay.json`
  with `"ready": true` (including `capture_variant`).
- `neurons/miner.py` encrypts both files, uploads to S3, attaches
  `ScreenReplayUAV`, then resets `ready` to false.
- Extra submissions while one is pending go into `queue/` automatically.

## `screen_replay.json` fields

| Field | Meaning |
|---|---|
| `ready` | `true` when queued for the miner; flipped to `false` after submit. |
| `photo_path` | Staged **FACE CLOSE-UP** (photo or video). |
| `photo_path_2` | Staged **ENVIRONMENT** still. |
| `capture_variant` | One of the five variety tracks above. |
| `primary_media` | `photo` or `video`. |
| `seed_image` | Pool / daily seed filename you started from. |
| `date` | Capture date `YYYY-MM-DD` (UTC). |
| `camera_used` | Camera/phone used for the capture. |
| `device_photographed` | `phone` / `tablet` / `laptop` / `monitor` / `tv`. |
| cue bools | `moire_pixel_grid`, `screen_glare_hotspots`, `perspective_keystone_distortion`, `gamma_contrast_shift`, `edge_crop_cues`. |

## Rules / good to know

- **No daily limit.** Never submit a duplicate.
- The **screen capture** must be real. Seed edits (eyes closed / blink /
  smile / smile-and-blink video) may be synthetic; the photograph/video of
  the screen must not be.
- Environment shot is **always** a still photo.
- Skipping this entirely is fine.

## Command-line flags

```bash
# Photo variant
python MIID/miner/real_image_miner_guide/submit_real_photo.py \
  --variant synthetic_eyes_closed \
  --face closeup.jpg --env wide.jpg \
  --seed-image 034633750981_f_doc.png \
  --camera "iPhone 15 Pro" --device phone \
  --moire --glare

# Video variant
python MIID/miner/real_image_miner_guide/submit_real_photo.py \
  --variant synthetic_video_blinking \
  --face replay.mp4 --env wide.jpg \
  --seed-image 034633750981_f_doc.png \
  --camera "iPhone 15 Pro" --device laptop
```

| Flag | Meaning |
|---|---|
| `--variant` | `device_camera` / `synthetic_eyes_closed` / `synthetic_video_blinking` / `synthetic_video_smiling` / `synthetic_video_smile_and_blink` |
| `--face` / `--env` | Inbox filenames for primary + environment |
| `--seed-image` | Pool / seed filename |
| `--camera` / `--device` / `--date` | Capture metadata |
| `--moire` / `--glare` / `--keystone` / `--gamma` / `--edge-crop` | Cue checklist |
