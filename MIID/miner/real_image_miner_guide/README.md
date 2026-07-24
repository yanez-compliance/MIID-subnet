# Real Screen-Replay Photo Submission

This folder is where you submit a **real, physical photograph** — not an
AI-generated one — of the daily seed image displayed on a real screen. This
is completely **optional** and is not required every round. There is **no
daily limit** — submit as many different real captures as you want, as
often as you like. The only hard rule is: **never submit a duplicate**
(the same capture/photos again) — that's filtered out and penalised. It's
also fine to skip this entirely, or only submit occasionally.

## TL;DR

1. Take TWO photos of the SAME capture, from two different angles/positions:
   display today's daily seed image on a real screen (phone, tablet, laptop,
   monitor, or TV) and photograph it twice with a **different** physical
   camera (no screenshots). Two angles of one capture, not two unrelated
   photos.
2. Drop both photo files into `inbox/` (this folder) — exactly 2 images.
3. Run:
   ```bash
   python MIID/miner/real_image_miner_guide/submit_real_photo.py
   ```
4. Answer the few questions it asks you (camera used, which device you
   photographed, which visual cues are visible). That's it — nothing else to
   do. Your miner process picks it up automatically and submits both photos
   as one submission the next time a validator queries you.
5. Want to submit again? Take a brand-new pair of photos and re-run the
   script — as many times as you like, whenever you're ready. Just make sure
   each pair is a genuinely new capture, never the same photos reused.

You do **not** need to restart your miner. `neurons/miner.py` checks
`screen_replay.json` on every incoming validator request, so as soon as the
script above finishes, the very next query will submit your photo.

## How it works under the hood

- `submit_real_photo.py` finds the two images you placed in `inbox/`, moves
  them into `staged/` (so `inbox/` is free for next time), and rewrites
  `screen_replay.json` with both photos' local paths + your answers, ending
  with `"ready": true`. It does **not** upload anything to S3 itself —
  that only happens once, below, at the real submission step.
- Your always-running miner process (`neurons/miner.py`) checks
  `screen_replay.json` every time it handles a validator request. When it
  sees `"ready": true`, it:
  1. Reads both photos from `photo_path` and `photo_path_2`.
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

## `screen_replay.json` fields

| Field | Meaning |
|---|---|
| `ready` | Set to `true` by `submit_real_photo.py` when a capture is queued. The miner flips it back to `false` after submitting. **You shouldn't normally need to edit this by hand.** |
| `photo_path` | Absolute path to the staged photo file (angle 1). |
| `photo_path_2` | Absolute path to the staged photo file (angle 2 — a different angle of the same capture). |
| `date` | Capture date, `YYYY-MM-DD` (UTC). Defaults to today. |
| `camera_used` | The camera/phone you used to take the photos, e.g. `"iPhone 15 Pro"`. |
| `device_photographed` | Which device displayed the seed image: one of `phone`, `tablet`, `laptop`, `monitor`, `tv`. |
| `moire_pixel_grid` | `true`/`false` — is a moiré/pixel-grid interference pattern visible? |
| `screen_glare_hotspots` | `true`/`false` — are specular glare hotspots visible? |
| `perspective_keystone_distortion` | `true`/`false` — is there off-angle/keystone distortion? |
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
- **Each submission needs exactly TWO photos** — two different angles of the
  *same* capture, not one photo and not two unrelated photos. This is basic
  proof it's a real physical photo, not a single static image reused twice.
- **The face must dominate both photos.** Frame so the face on the screen is
  the main subject — large enough for reliable face detection and still
  matchable to the seed identity. Don't crop so tight that the screen
  disappears, and don't pull so far back that the face is tiny.
- It's a **real physical photo**, not a screenshot and not AI-generated. Do
  not use FLUX or any generator for this — that defeats the purpose.
- `inbox/` must contain exactly 2 images when you run the script — the
  script errors out if it finds fewer or more than 2, so there's no
  ambiguity about which pair belongs together.
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
