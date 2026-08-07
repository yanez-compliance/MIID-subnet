# Real Screen-Replay Photo Submission

This folder is where you submit a **real, physical photograph** — not an
AI-generated one — of a seed image displayed on a real screen. This is
completely **optional** and is not required every round. There is **no
daily limit** — submit as many different real captures as you want, as
often as you like. The only hard rule is: **never submit a duplicate**
(the same capture/photos again) — that's filtered out and penalised. It's
also fine to skip this entirely, or only submit occasionally.

> **Sandbox mode (current):** the validator isn't sending you a seed image
> right now. Instead, pick any ONE image yourself, at random, from
> `MIID/validator/fixed_image/` — a small pool that ships with this repo (no
> download needed). This lets you practice the flow before the validator
> resumes pushing a seed image every round.

## TL;DR

1. Pick ONE image at random from `MIID/validator/fixed_image/` in this repo.
2. Take TWO photos of the SAME capture with a **different** physical camera
   (no screenshots):
   - **Photo 1 — FACE CLOSE-UP:** face on screen is dominant and **centered**,
     with **as little angular/perspective distortion as possible** (near
     head-on to the display). Keep a little screen/bezel context — don't
     crop the face alone.
   - **Photo 2 — ENVIRONMENT:** wider shot of the **whole screen/device in
     its surroundings** (desk, room, laptop keyboard, etc.). Angular
     distortion, keystone, glare, and moiré are fine here; the face should
     still be visible on the screen so the shot links to photo 1.
3. Drop both photo files into `inbox/` (this folder) — exactly 2 images.
   iPhone HEIC/HEIF files are fine: the script converts them to JPEG
   automatically when it runs (needs `pillow-heif`, already in
   `requirements.txt`).
4. Run:
   ```bash
   python MIID/miner/real_image_miner_guide/submit_real_photo.py
   ```
5. Tell the script which file is the face close-up vs the environment shot,
   then answer the few questions it asks (which pool image you used, camera
   used, which device you photographed, which visual cues are visible).
   That's it — nothing else to do. Your miner process picks it up
   automatically and submits both photos as one submission the next time a
   validator queries you.
6. Want to submit again? Pick a (possibly different) pool image, take a
   brand-new pair of photos, and re-run the script — as many times as you
   like, whenever you're ready. Just make sure each pair is a genuinely new
   capture, never the same photos reused. If your previous capture hasn't
   gone out yet, it's queued automatically (see "Submitting more than one at
   a time" below) — nothing is ever lost or overwritten.

You do **not** need to restart your miner. `neurons/miner.py` checks
`screen_replay.json` on every incoming validator request, so as soon as the
script above finishes, the very next query will submit your photo.

## How it works under the hood

- `submit_real_photo.py` finds the two images you placed in `inbox/`, asks
  which is the face close-up vs environment shot, moves them into `staged/`
  (so `inbox/` is free for next time), and rewrites `screen_replay.json`
  with both photos' local paths + your answers, ending with `"ready": true`.
  It does **not** upload anything to S3 itself — that only happens once,
  below, at the real submission step.
- Your always-running miner process (`neurons/miner.py`) checks
  `screen_replay.json` every time it handles a validator request. When it
  sees `"ready": true`, it:
  1. Reads both photos from `photo_path` (face) and `photo_path_2` (env).
  2. Encrypts each with drand timelock (same as every other submission).
  3. Uploads both **encrypted** photos to the real S3 submissions path (the
     only S3 uploads in this whole flow).
  4. Sends both references, as one submission, to the validator that
     queried it.
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
   captures back-to-back as you want (take photos, run the script, repeat)
   and each one will eventually be sent, in the order you submitted them.

You don't need to do anything special to use this — it kicks in
automatically whenever you submit while a previous capture is still
pending. `queue/` is just a holding area; you shouldn't normally need to
look inside it.

## `screen_replay.json` fields

| Field | Meaning |
|---|---|
| `ready` | Set to `true` by `submit_real_photo.py` when a capture is queued. The miner flips it back to `false` after submitting. **You shouldn't normally need to edit this by hand.** |
| `photo_path` | Absolute path to the staged **FACE CLOSE-UP** (centered, minimal angular distortion). |
| `photo_path_2` | Absolute path to the staged **ENVIRONMENT** shot (whole screen/device in surroundings). |
| `seed_image` | Filename of the `MIID/validator/fixed_image/` pool image you randomly picked and photographed (sandbox mode). |
| `date` | Capture date, `YYYY-MM-DD` (UTC). Defaults to today. |
| `camera_used` | The camera/phone you used to take the photos, e.g. `"iPhone 15 Pro"`. |
| `device_photographed` | Which device displayed the seed image: one of `phone`, `tablet`, `laptop`, `monitor`, `tv`. |
| `moire_pixel_grid` | `true`/`false` — is a moiré/pixel-grid interference pattern visible? |
| `screen_glare_hotspots` | `true`/`false` — are specular glare hotspots visible? |
| `perspective_keystone_distortion` | `true`/`false` — is there off-angle/keystone distortion? (often on the environment shot) |
| `gamma_contrast_shift` | `true`/`false` — is there a colour/brightness shift typical of a display capture? |
| `edge_crop_cues` | `true`/`false` — are screen borders/bezel/cropping visible? |

Report the cue checklist honestly — reviewers verify it, and a real photo
may legitimately show anywhere from 0 to 5 of these cues.

## Rules / good to know

- **No daily limit — send as many captures as you want.** Submit as often as
  you like; there's no cap on how many real screen-replay submissions you
  send.
- **Never submit a duplicate.** Every submission must be a genuinely new
  capture. Re-running the script on the same photos, or reusing photos from
  a previous submission, will be detected and penalised.
- **Each submission needs exactly TWO photos** of the *same* capture:
  - Photo 1: face-dominant, centered, **minimal angular distortion**
  - Photo 2: whole screen/device **environment** (distortion OK)
  Not one photo, and not two unrelated photos.
- It's a **real physical photo**, not a screenshot and not AI-generated. Do
  not use FLUX or any generator for this — that defeats the purpose.
- `inbox/` must contain exactly 2 images when you run the script — the
  script errors out if it finds fewer or more than 2, so there's no
  ambiguity about which pair belongs together. You'll be asked which file
  is the face close-up vs the environment shot.
- Skipping this entirely is fine. It's optional, not scored every round.

## Command-line flags (optional, for scripting/automation)

You can skip all the interactive prompts by passing flags up front:

```bash
python MIID/miner/real_image_miner_guide/submit_real_photo.py \
  --face closeup.jpg --env wide.jpg \
  --seed-image 034633750981_f_doc.png \
  --camera "iPhone 15 Pro" \
  --device phone \
  --moire --glare
```

| Flag | Meaning |
|---|---|
| `--face TEXT` | Inbox filename of the FACE CLOSE-UP |
| `--env TEXT` | Inbox filename of the ENVIRONMENT shot |
| `--seed-image TEXT` | Filename of the `fixed_image/` pool image you used |
| `--camera TEXT` | Camera/phone used to take the photo |
| `--device {phone,tablet,laptop,monitor,tv}` | Device the seed image was displayed on |
| `--date YYYY-MM-DD` | Capture date (defaults to today, UTC) |
| `--moire` / `--glare` / `--keystone` / `--gamma` / `--edge-crop` | Mark a visual cue as visible (omit if not visible) |
