# Cycle Scoring — KAV, UAV & Rewards

**Goal for Phase 5 Cycle 2:** find vulnerabilities in **deepfake detection** and **passive liveness**, and break them, so those models can be improved.

Face variations are aimed at deepfake detection. Screen replay is aimed at passive liveness. Face grade (`POST /face_grade`) and Screen replay grade (`POST /screen_replay_grade`) tell a miner whether the image broke and what the grade is.

The live validator grade is `POST /grade_v3`. What you send and what comes back is the same shape as `/grade_v2`: one random face slot, then a validation score and an identity score. The score and the comment inside that result are the new rubric on this page. Screen replay is stored with the round and is not part of that live sample.

**This section is Sandbox** (Nov 19, 2026 onward; Phase 5 started Sep 24, 2026). Sandbox does **not** collect UAV. Rewards are **KAV only**. Reviewers can still finish UAV from the previous cycle. The mix below is how UAV is scored once it is collected.

---

## Current cycle: Phase 5, Cycle 2 — Sandbox

Live **KAV** grades one sampled face image each round. Voice and screen replay are stored with the round. They are not part of the Sandbox reward.

---

## What a round asks for

| Piece | Count | Share of UAV |
|-------|------:|-------------:|
| Face variations | 5 | **50%** (10% each) |
| Screen replay | 1 | **25%** |
| Voice clips | 5 | **25%** (one average) |

Face types, always: `background_in`, `background_out`, `lighting_edit+expression_edit`, `lighting_edit+pose_edit`, `pose_edit+expression_edit`.

Voice types, always: `neutral`, `surprised`, `excited`, `angry`, `whisper`. Each clip also gets a random background-noise level: `none`, `light`, or `high`. The emotion is not scaled. Every clip speaks the same challenge text.

Screen replay is one image aimed at Passive Liveness v3, with the named device and the requested cues.

---

## How the UAV score is built

Faces are no longer averaged into one number. Each face keeps its own score and its own delta, so one image that breaks Deepfake V1 is not lost in the other four.

```
each face        → score 0–5 → that slot's delta → 10% of UAV
screen replay    → score 0–5 → one delta         → 25% of UAV
five voice clips → average   → one delta         → 25% of UAV
```

A face or a screen replay scores **5** only when the detector breaks and the ask was met (identity held, the edit or the device and cues are there). If the detector holds, that score stops at **3**.

Voice has no gate. Each clip is scored out of 5 (tokens match the prompt, SpeechBrain similarity at least 0.6, and the named style is what you hear). The five scores are averaged, rounded half up, and that one number is looked up in the delta table.

### Delta

| Score | When | Δrep |
|------:|------|-----:|
| +5 | Breaks the detector, and the ask was met | +0.10 |
| +4 | Breaks, but the edit or the match is only near perfect | +0.04 |
| +3 | Best score if the detector holds | +0.03 |
| +2 | Okay | +0.02 |
| +1 | Decent | +0.01 |
| 0 | Unclear | 0.00 |
| −1 | Partial spam | −0.05 |
| −2 | Misleading | −0.08 |
| −3 | Duplicate | −0.10 |
| −4 | Not enough info | −0.25 |
| −5 | Cheat | −0.50 |

Nothing scores 6. The old +0.15 row is unused.

### Voice background noise

Picked at random per clip, the same way a face variation gets an intensity (`light`, `medium`, or `far`).

| Level | What you hear |
|-------|----------------|
| `none` | No background noise |
| `light` | Light background noise, speech still easy to hear |
| `high` | High background noise, speech still intelligible |

---

## Preview your score

Two calls, both at `https://api.yanez.ai:5000`. The response is `breaks`, a `score` from -5 to 5, and a `comment`.

| What you want | Run | Send |
|---------------|-----|------|
| Did it break? | `miner_api/miner_api.py` | The picture only |
| Exact grade | `miner_api/exact_grade.py` | The picture, the seed, and the prompt |

| Call | Detector | Exact-grade prompt |
|------|----------|--------------------|
| `POST /face_grade` | Deepfake | Variation type and intensity |
| `POST /screen_replay_grade` | Passive Liveness | The cues the challenge asked for |

Client and request shape: [miner_api/api_white_paper.md](../miner_api/api_white_paper.md).

---

## Emission breakdown

Normal weight split (what validators set on-chain):

| Destination | Share | Notes |
|-------------|-------|--------|
| Data-generation miners (KAV + UAV) | **~50%** | Burn removed |
| Dual-incentive / partner pool | **~50%** | Partner hotkey (UID **14** on mainnet) |
| Burn | **~0%** | Burn UID only as fallback / unused KAV |

Within the **50% miner share**:

| Component | Weight of miner share | Of total emissions (approx.) |
|-----------|----------------------|------------------------------|
| **KAV** (live image quality) | **10%** | **~5%** |
| **UAV** (reputation / post-validation) | **90%** | **~45%** |

With the defaults shipped in this repo, most of the miner share still comes from the **existing reputation cache**, not from this round's new voice or screen-replay files.

If the reputation cache is empty, that UAV share is rerouted to the dual-incentive partner. If the partner hotkey is absent from the metagraph, the 50% partner fraction falls back to the burn UID.

---

## KAV vs UAV

| | **KAV** | **UAV** |
|---|---------|---------|
| **What it is** | Live, per-round image quality score | Reputation from post-validation |
| **When paid** | Continuously on-chain during the cycle | Finalized after review; paid especially in **Execution** |
| **What is graded** | One sampled face image (`/grade_v3`, same shape as `/grade_v2`) | Each face, the screen replay, and the voice average |
| **Weight** | 10% of miner share | 90% of miner share |

Live KAV grades **one** of the five face-variation slots, chosen at random that round, via `/grade_v3`. The request and the result match `/grade_v2`. The score and comment use the new rubric above. Identity preservation is still the tiebreaker and a gate (**≥ 0.6** among the top 50). Screen replay and voice are not part of live KAV.

When UAV grading is enabled, the validator combines `kav_weight × Q` with `uav_weight × (normalized_rep_score × tier_multiplier)`, then rescales so the miner pool hits the configured targets. Unqueried miners can still receive UAV-only rewards from their reputation snapshot. New miners not in the snapshot get KAV only.

---

## Sandbox and Execution

### Sandbox (this section)

- Miners try to break deepfake detection and passive liveness, and try the five voice styles and noise levels.
- Rewards are **KAV only**. UAV for this cycle is not collected, and prediction scores do not increase.
- Reviewers use this time to finish grades from the previous cycle.

### Execution

- Finalized UAV from the previous cycle is distributed.
- UAV prediction for the current cycle accumulates.
- Live KAV continues.

---

## Quick reference

```
Emissions (weights)
├── ~50% data-generation miners
│     ├── 10% KAV  → ~5% of total  (live image grading each round)
│     └── 90% UAV  → ~45% of total (post-validation reputation; not collected in Sandbox)
├── ~50% dual-incentive / partner pool (UID 14)
└── ~0% burn

Sandbox (now)
└── KAV only
      └── one sampled face image via /grade_v3
            same result shape as /grade_v2
            score and comment use the new rubric
            identity gate ≥ 0.6 among the top 50

UAV mix (scored once UAV is collected)
├── 50% faces — five slots, each with its own delta
│     ├── 10% background_in
│     ├── 10% background_out
│     ├── 10% lighting + expression
│     ├── 10% lighting + pose
│     └── 10% pose + expression
│           score 0–5
│           5 only if Deepfake breaks, identity held, and the edit is there
│           otherwise at most 3
├── 25% screen replay → one delta
│     └── score 0–5
│           5 only if Passive Liveness breaks (device, cues, identity)
│           otherwise at most 3
└── 25% voice → average the five clips, round half up, one delta
      ├── neutral, surprised, excited, angry, whisper
      ├── each clip: random noise none / light / high
      └── each clip out of 5 (tokens, SpeechBrain ≥ 0.6, named style)
            no break gate

Goal
├── Face variations  → break deepfake detection
└── Screen replay    → break passive liveness
```

---

## Related docs

- [Miner guide](miner.md)
- [Validator guide](validator.md)
- [Example score sheet](uav-score.html)
- Implementation: `MIID/validator/reward.py`, `MIID/validator/voice_variations.py`, `MIID/validator/forward.py`
