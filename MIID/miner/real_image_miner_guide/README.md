# Real Screen-Replay Photo Submission

This folder is where you submit a **real, physical capture** — photograph or
video of a seed displayed on a real screen. Optional; **no daily limit**;
**never submit a duplicate**.

> Validators send **today's and tomorrow's image-of-the-day** with every
> query. Your miner saves them to `seeds/` (see `seeds.json`). Photograph
> **today's** seed for captures you submit now; tomorrow's is sent early so
> you can prepare overnight.

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

1. Display **today's IOTD** from `seeds/` (or tomorrow's if you are
   prepping ahead). Optionally edit it per variant above.
2. Display it on a real screen; capture with a different camera:
   - **FACE CLOSE-UP** — photo (1–3) or video (4–6)
   - **ENVIRONMENT** — wider still
3. Drop into `inbox/` (2 images, or 1 video + 1 image for video variants).
4. Run:
   ```bash
   python MIID/miner/real_image_miner_guide/submit_real_photo.py
   ```
5. Answer the prompts (variant, which IOTD — today or tomorrow, camera,
   device photographed, which visual cues are visible). That's it — your
   miner picks it up automatically and submits both files as one
   submission the next time a validator queries you.
6. Want to submit again? Take a brand-new pair of today's (or tomorrow's)
   seed and re-run the script — as many times as you like. Each pair must
   be a genuinely new capture, never the same media reused. If a
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
| `--seed-image` | IOTD filename you photographed (today or tomorrow) |
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

`screen_replay.json` only has **one active slot**. Captures are sent **FIFO
(oldest first)**:

1. The first capture occupies `screen_replay.json` until the miner sends it.
2. If you run `submit_real_photo.py` again while that slot is still `ready`,
   the **new** capture is appended to `queue/` (timestamped). The waiting
   capture in `screen_replay.json` is not bumped.
3. After each successful send, the miner promotes the oldest **due** queued
   capture into the active slot.

**Tomorrow's image-of-the-day:** if you photograph tomorrow's seed, `date`
is set to tomorrow's UTC date and `seed_slot` is `"tomorrow"`. The miner
will **not** upload that capture until that UTC day. It stays in `queue/`
(or is skipped in the active slot) so today's captures can still go out.

You don't need to do anything special — queueing and the date hold kick in
automatically. `queue/` is just a holding area.

## `screen_replay.json` fields

| Field | Meaning |
|---|---|
| `ready` | Set to `true` by `submit_real_photo.py` when a capture is queued. The miner flips it back to `false` after submitting. **You shouldn't normally need to edit this by hand.** |
| `photo_path` | Absolute path to the staged face close-up (photo for variants 1–3, video for 4–6). **No spaces** — the submit script sanitizes the filename. |
| `photo_path_2` | Absolute path to the staged environment still of the same capture. **No spaces.** |
| `seed_image` | Filename of the IOTD you photographed (today or tomorrow; see `seeds/`). |
| `date` | IOTD date, `YYYY-MM-DD` (UTC). **Tomorrow's seed uses tomorrow's date** — the miner will not upload until that day. |
| `seed_slot` | `today` or `tomorrow`, from the seed you picked. |
| `camera_used` | The camera/phone you used to take the capture, e.g. `"iPhone 15 Pro"`. |
| `device_photographed` | Which device displayed the seed: one of `phone`, `tablet`, `laptop`, `monitor`, `tv`. |
| `capture_variant` | One of the six variant keys above. Photo vs video is implied by this (seed_video_* → video). |
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
