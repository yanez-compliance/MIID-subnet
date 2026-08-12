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
5. Answer the prompts (variant, which pool image, camera, device photographed,
   which visual cues are visible). That's it — your miner picks it up
   automatically and submits both files as one submission the next time a
   validator queries you.
6. Want to submit again? Pick a (possibly different) pool image, take a
   brand-new pair, and re-run the script — as many times as you like. Each
   pair must be a genuinely new capture, never the same media reused. If a
   previous capture hasn't gone out yet, it's queued automatically (see
   "Submitting more than one at a time" below).

You do **not** need to restart your miner. `neurons/miner.py` checks
`screen_replay.json` on every incoming validator request, so as soon as the
script finishes, the very next query will submit your capture.

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

## How it works under the hood

- `submit_real_photo.py` finds the face close-up + environment pair in
  `inbox/`, moves them into `staged/` (so `inbox/` is free for next time),
  and rewrites `screen_replay.json` with both local paths + your answers,
  ending with `"ready": true`. It does **not** upload anything to S3 itself —
  that only happens once, below, at the real submission step.
- Your always-running miner process (`neurons/miner.py`) checks
  `screen_replay.json` every time it handles a validator request. When it
  sees `"ready": true`, it:
  1. Reads both files from `photo_path` (face close-up photo or video) and
     `photo_path_2` (environment still).
  2. Encrypts each with drand timelock (same as every other submission).
  3. Uploads both **encrypted** files to the real S3 submissions path (the
     only S3 uploads in this whole flow).
  4. Sends both references, as one submission, to the validator that
     queried it (including `capture_variant` and cue metadata).
  5. Resets `screen_replay.json` back to a blank, `"ready": false` state so
     it won't accidentally resubmit the same capture again. You're free to
     queue up a brand-new capture right away.
- If `"ready"` is `false` (the normal, default state), the miner does
  nothing extra — this is expected most rounds. It's optional either way.

## Submitting more than one at a time (the `queue/` folder)

`screen_replay.json` only has **one active slot**. If you run
`submit_real_photo.py` again before your always-running miner process has
had a chance to pick up and send the previous capture (i.e. `ready` is still
`true`), nothing is lost:

1. `submit_real_photo.py` notices the active slot is still occupied, copies
   that pending capture into `queue/` (named with a timestamp so order is
   preserved), and writes your **new** capture into `screen_replay.json` —
   your new one takes the active slot immediately.
2. Every time the miner process finishes sending the active capture, it
   checks `queue/` and automatically promotes the **oldest** queued capture
   into the now-empty active slot, ready to go out on the next validator
   query.
3. This repeats until `queue/` is empty — so you can queue up as many
   captures back-to-back as you want (take captures, run the script, repeat)
   and each one will eventually be sent, in the order you submitted them.

You don't need to do anything special to use this — it kicks in
automatically whenever you submit while a previous capture is still
pending. `queue/` is just a holding area; you shouldn't normally need to
look inside it.

## `screen_replay.json` fields

| Field | Meaning |
|---|---|
| `ready` | Set to `true` by `submit_real_photo.py` when a capture is queued. The miner flips it back to `false` after submitting. **You shouldn't normally need to edit this by hand.** |
| `photo_path` | Absolute path to the staged face close-up (photo for variants 1–3, video for 4–6). |
| `photo_path_2` | Absolute path to the staged environment still of the same capture. |
| `seed_image` | Filename of the `MIID/validator/fixed_image/` pool image you randomly picked and photographed (sandbox mode). |
| `date` | Capture date, `YYYY-MM-DD` (UTC). Defaults to today. |
| `camera_used` | The camera/phone you used to take the capture, e.g. `"iPhone 15 Pro"`. |
| `device_photographed` | Which device displayed the seed: one of `phone`, `tablet`, `laptop`, `monitor`, `tv`. |
| `capture_variant` | One of the six variant keys above. |
| `primary_media` | `photo` or `video` — set automatically from the variant. |
| `moire_pixel_grid` | `true`/`false` — is a moiré/pixel-grid interference pattern visible? |
| `screen_glare_hotspots` | `true`/`false` — are specular glare hotspots visible? |
| `perspective_keystone_distortion` | `true`/`false` — is there off-angle/keystone distortion? |
| `gamma_contrast_shift` | `true`/`false` — is there a colour/brightness shift typical of a display capture? |
| `edge_crop_cues` | `true`/`false` — are screen borders/bezel/cropping visible? |

Report the cue checklist honestly — reviewers verify it, and a real capture
may legitimately show anywhere from 0 to 5 of these cues. Keystone / glare /
moiré are often more obvious on the environment shot.

## Rules / good to know

- **No daily limit — send as many captures as you want.** Submit as often as
  you like; there's no cap on how many real screen-replay submissions you
  send.
- **Never submit a duplicate.** Every submission must be a genuinely new
  capture. Re-running the script on the same media, or reusing files from a
  previous submission, will be detected and penalised.
- **Each submission needs exactly two files** — a face close-up (photo or
  video, depending on variant) plus a distinct environment still of the
  *same* physical setup. Not one file, and not two unrelated captures.
- **The face must dominate the close-up.** Frame so the face on the screen is
  the main subject — large enough for reliable face detection and still
  matchable to the seed identity. Don't crop so tight that the screen
  disappears, and don't pull so far back that the face is tiny. The
  environment shot should show the whole device/scene.
- It's a **real physical capture**, not a screenshot and not AI-generated.
  Do not use FLUX or any generator for this — that defeats the purpose.
- `inbox/` must contain the right pair when you run the script: **2 images**
  for photo variants, or **1 video + 1 image** for video variants. The
  script errors out if the count/types don't match.
- Skipping this entirely is fine. It's optional, not scored every round.
