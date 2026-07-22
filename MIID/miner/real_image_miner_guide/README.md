# Real Screen-Replay Photo Submission

This folder is where you submit a **real, physical photograph** — not an
AI-generated one — of the daily seed image displayed on a real screen. This
is completely **optional** and is not required every round (or every day).
It's fine to skip it entirely, or only submit occasionally.

## TL;DR

1. Take a photo: display today's daily seed image on a real screen (phone,
   tablet, laptop, monitor, or TV) and photograph it with a **different**
   physical camera (no screenshots).
2. Drop that photo file into `inbox/` (this folder).
3. Run:
   ```bash
   python MIID/miner/real_image_miner_guide/submit_real_photo.py
   ```
4. Answer the few questions it asks you (camera used, which device you
   photographed, which visual cues are visible). That's it — nothing else to
   do. Your miner process picks it up automatically and submits it the next
   time a validator queries you.

You do **not** need to restart your miner. `neurons/miner.py` checks
`screen_replay.json` on every incoming validator request, so as soon as the
script above finishes, the very next query will submit your photo.

## How it works under the hood

- `submit_real_photo.py` finds the image you placed in `inbox/`, moves it
  into `staged/` (so `inbox/` is free for next time), and rewrites
  `screen_replay.json` with the photo's local path + your answers, ending
  with `"ready": true`. It does **not** upload anything to S3 itself —
  that only happens once, below, at the real submission step.
- Your always-running miner process (`neurons/miner.py`) checks
  `screen_replay.json` every time it handles a validator request. When it
  sees `"ready": true`, it:
  1. Reads the photo from `photo_path`.
  2. Encrypts it with drand timelock (same as every other submission).
  3. Uploads the **encrypted** photo to the real S3 submissions path (the
     only S3 upload in this whole flow).
  4. Sends the reference to the validator that queried it.
  5. Resets `screen_replay.json` back to a blank, `"ready": false` state so
     it won't accidentally resubmit the same photo again.
- If `"ready"` is `false` (the normal, default state), the miner does
  nothing extra — this is expected most rounds. It's optional either way.

## `screen_replay.json` fields

| Field | Meaning |
|---|---|
| `ready` | Set to `true` by `submit_real_photo.py` when a photo is queued. The miner flips it back to `false` after submitting. **You shouldn't normally need to edit this by hand.** |
| `photo_path` | Absolute path to the staged photo file. |
| `date` | Capture date, `YYYY-MM-DD` (UTC). Defaults to today. |
| `camera_used` | The camera/phone you used to take the photo, e.g. `"iPhone 15 Pro"`. |
| `device_photographed` | Which device displayed the seed image: one of `phone`, `tablet`, `laptop`, `monitor`, `tv`. |
| `moire_pixel_grid` | `true`/`false` — is a moiré/pixel-grid interference pattern visible? |
| `screen_glare_hotspots` | `true`/`false` — are specular glare hotspots visible? |
| `perspective_keystone_distortion` | `true`/`false` — is there off-angle/keystone distortion? |
| `gamma_contrast_shift` | `true`/`false` — is there a colour/brightness shift typical of a display capture? |
| `edge_crop_cues` | `true`/`false` — are screen borders/bezel/cropping visible? |

Report the cue checklist honestly — reviewers verify it, and a real photo
may legitimately show anywhere from 0 to 5 of these cues.

## Rules / good to know

- **At most one submission per UTC day.** Submitting more than once per day
  will not help and may be penalised — don't re-run the script again the
  same day just because you feel like it.
- It's a **real physical photo**, not a screenshot and not AI-generated. Do
  not use FLUX or any generator for this — that defeats the purpose.
- If you place more than one image in `inbox/`, the script picks the first
  one alphabetically and warns you — keep only one photo in there at a time.
- Skipping this entirely is fine. It's optional, not scored every round.

## Command-line flags (optional, for scripting/automation)

You can skip all the interactive prompts by passing flags up front:

```bash
python MIID/miner/real_image_miner_guide/submit_real_photo.py \
  --camera "iPhone 15 Pro" \
  --device phone \
  --moire --glare
```

| Flag | Meaning |
|---|---|
| `--camera TEXT` | Camera/phone used to take the photo |
| `--device {phone,tablet,laptop,monitor,tv}` | Device the seed image was displayed on |
| `--date YYYY-MM-DD` | Capture date (defaults to today, UTC) |
| `--moire` / `--glare` / `--keystone` / `--gamma` / `--edge-crop` | Mark a visual cue as visible (omit if not visible) |
