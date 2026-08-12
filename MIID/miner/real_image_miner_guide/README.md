# Real Screen-Replay Photo Submission

This folder is where you submit a **real, physical capture** — photograph or
video of a seed displayed on a real screen. Optional; **no daily limit**;
**never submit a duplicate**.

> **Sandbox mode (current):** pick one image at random from
> `MIID/validator/fixed_image/`.

## Capture variants (pick one)

How you prepared the **seed** before photographing/recording the screen.
Device and camera are asked separately.

| `#` | `capture_variant` | Meaning | Primary |
|---|---|---|---|
| 1 | `seed_unchanged` | Seed as-is | Photo |
| 2 | `seed_smiling` | Seed edited to smile | Photo |
| 3 | `seed_eyes_closed` | Seed edited to eyes closed | Photo |
| 4 | `seed_video_blinking` | Blink seed-video on screen | Video |
| 5 | `seed_video_smiling` | Smile seed-video on screen | Video |
| 6 | `seed_video_smile_and_blink` | Smile + blink seed-video on screen | Video |

Every variant also needs an **environment still** of the whole device/scene.

## How the two files are reviewed (cross-view consistency)

Manual review / automated flags check that close-up + environment are the
**same physical capture**:

- Same identity / seed face on the screen in both shots
- Same device / bezel geometry
- Roughly the same lighting and glare direction
- Distinct file hashes (do not upload the same file twice)

Capture both back-to-back of the same setup. Do not pair unrelated photos
or paste a new seed into an old capture shell.

## TL;DR

1. Pick a pool seed; optionally edit it per variant above.
2. Display it on a real screen; capture with a different camera:
   - **FACE CLOSE-UP** — photo (1–3) or video (4–6)
   - **ENVIRONMENT** — wider still
3. Drop into `inbox/` (2 images, or 1 video + 1 image for video variants).
4. Run:
   ```bash
   python MIID/miner/real_image_miner_guide/submit_real_photo.py
   ```

## Command-line examples

```bash
python MIID/miner/real_image_miner_guide/submit_real_photo.py \
  --variant seed_smiling \
  --face closeup.jpg --env wide.jpg \
  --seed-image 034633750981_f_doc.png \
  --camera "iPhone 15 Pro" --device phone

python MIID/miner/real_image_miner_guide/submit_real_photo.py \
  --variant seed_video_blinking \
  --face replay.mp4 --env wide.jpg \
  --seed-image 034633750981_f_doc.png \
  --camera "iPhone 15 Pro" --device laptop
```

| Flag | Meaning |
|---|---|
| `--variant` | One of the six keys above |
| `--face` / `--env` | Inbox filenames for primary + environment |
| `--seed-image` | Pool / seed filename |
| `--camera` / `--device` / `--date` | Capture metadata |
| `--moire` / `--glare` / `--keystone` / `--gamma` / `--edge-crop` | Cue checklist |
