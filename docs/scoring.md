# Cycle Scoring — KAV, UAV & Rewards

This document explains how miner scoring and emissions work on the Yanez subnet (Subnet 54): the **Sandbox / Execution** cycle structure, **KAV vs UAV**, and how **Phase 5 Cycle 1 Execution** builds UAV prediction scores from face variations, screen replays, and voice — including the point totals and primary grade bands reviewers use.

---

## Current cycle: Phase 5, Cycle 1 — Execution

**Status:** Phase 5, Cycle 1 is in the **Execution** section (Oct 19 – Nov 19, 2026).

### What we’re looking for this round

The main focus this round is **voice**. UAV prediction scoring for this cycle is split evenly:

| Component | Share of final UAV score | Notes |
|-----------|--------------------------|--------|
| **Voice clones** | **50%** | Primary focus — identity preservation + spoken-word match |
| **Face UAV** | **50%** | Split evenly inside face: **50%** variation edits + **50%** screen replay (~25% + ~25% of final UAV) |

Live **KAV** continues to grade face image submissions online (as in previous cycles). Voice is collected for **UAV post-grading only** — it does not receive a live KAV grade.

Finalized UAV rewards from the previous cycle (Phase 4, Cycle 5) are distributed during this Execution section, while UAV prediction scores accumulate for Phase 5, Cycle 1.

---

## How UAV prediction scoring works (this cycle)

UAV scores are built from **manual / post-validation review**, not from the live KAV grading API. Reviewers score submissions, apply a **delta** against the live prediction, and fold the result into your UAV reputation (age factors may also apply).

### Point totals

| Block | Points | Notes |
|-------|--------|--------|
| **Face variations** (5 images) | **Out of 25** | Up to **5 points** each |
| **Screen replay** (1 image) | **Out of 6** | Up to **6 points** |
| **Voice** (1 clip) | **Out of 5** | Up to **5 points** |

### Pipeline

```
face_variations  → out of 25  → apply delta → face_var'
screen_replay    → out of 6   → apply delta → screen'
face_UAV         ← average( face_var' , screen' )

voice            → out of 5   → apply delta → voice_UAV

final_UAV        ≈ average( face_UAV , voice_UAV )
                   (* age and other factors)
```

Face UAV is **50%** of final UAV (half variations / half screen ≈ 25% + 25%). Voice is the other **50%**.

New miners with no reputation snapshot receive **KAV only** until they build UAV through validated submissions.

---

## Grade bands

Primary bands reviewers use. Cases outside these bands (failed identity, missing cues, wrong device, empty/corrupt audio, etc.) are scored below the partial band or rejected.

### Face variations — out of 25

Five images. Each can earn **up to 5 points**. Sum them for a total **out of 25**.

Scored on how well the requested variation was applied and how well **identity** is preserved.

Covers the non–screen-replay slots (`background_in` / `background_out`, combined lighting / expression / pose edits, and any forced-query mix):

| Variation | What it changes |
|-----------|-----------------|
| `background_in` | Indoor background swap |
| `background_out` | Outdoor background swap |
| `lighting_edit+expression_edit` | Lighting and expression together |
| `lighting_edit+pose_edit` | Lighting and pose together |
| `pose_edit+expression_edit` | Pose and expression together |

**Checklist (per image):**

- [ ] **Identity** is preserved relative to the seed
- [ ] The **prompted edit** was actually applied (right type / intensity, no extra unintended changes that break the ask)

> Five images × 5 = **25** max. That score out of 25, then delta, is **half** of face UAV.

### Screen replay — out of 6

One image. **Up to 6 points**. Screen replay is the other **half** of face UAV on its own.

Must be a face shown on a device screen (as requested), with the named device and required visual cues, and identity preserved vs the seed.

| Points | Breaks PL v3 (`/is_live`)? | Cues present? | Device correct? | Identity preserved? |
|-------:|:--------------------------:|:-------------:|:---------------:|:-------------------:|
| **6** | Yes | Yes | Yes | Yes |
| **3** | No | Yes | Yes | Yes |

**Checklist (all required for 6 or 3):**

- [ ] Visual **cues** we asked for are clearly present
- [ ] **Device** matches what was requested
- [ ] **Identity** is preserved relative to the seed
- [ ] Then: did it **break Passive Liveness v3**?
  - Yes → **6 points**
  - No → **3 points** (half)

> Breaking PL v3 is what separates full points from half. Cues, device, and identity alone are not enough for 6.

### Voice — out of 5

One clip. **Up to 5 points**. Based on speaker identity vs the reference (SpeechBrain) and spoken-token match to the prompt.

| Points | Token match | Identity (SpeechBrain) |
|-------:|-------------|------------------------|
| **5** | Spoken tokens **match** the prompt tokens | Similarity **≥ 0.6** |
| **3** | A **couple** of tokens mismatched | Identity still **preserved** |

**Checklist:**

- [ ] Clip speaks the challenge text (language `en` or `es`, including hotkey bookends when present)
- [ ] Speaker still matches the reference voice
- [ ] Then:
  - Tokens match **and** SpeechBrain ≥ **0.6** → **5 points**
  - A couple of tokens off, but identity still held → **3 points**

---

## Emission breakdown

> **Announcement — burn removed; miner rewards slightly up.**
> We are changing the subnet emission cut from **18% owner / 41% validator / 41% miner** to **18% owner / 49% validator / 33% miner**, and at the same time removing burn from the weight split (see below). This does **not** cut data-generation miner rewards — overall they rise slightly. The point of the change is lowering our burn rate.
>
> | | Old | New |
> |-|-----|-----|
> | Subnet cut (miner share of total emissions) | 41% | 33% |
> | Weight to data-gen miners | 35% of miner emissions | 50% of miner emissions |
> | Effective data-gen take of **total** emissions | 41% × 35% ≈ **14.35%** | 33% × 50% = **16.5%** |
>
> Same story for the dual-incentive partner pool (also 14.35% → 16.5% of total). Former burn (~12.3% of total) goes to zero.

Normal weight split (what validators set on-chain):

| Destination | Share | Notes |
|-------------|-------|--------|
| Data-generation miners (KAV + UAV) | **~50%** | Up from ~35%; burn removed |
| Dual-incentive / partner pool | **~50%** | Partner hotkey (UID **14** on mainnet) |
| Burn | **~0%** | Removed (burn UID only as fallback / unused KAV) |

Within the **50% miner share**:

| Component | Weight of miner share | Of total emissions (approx.) |
|-----------|----------------------|------------------------------|
| **KAV** (live image quality) | **10%** | **~5%** |
| **UAV** (reputation / post-validation) | **90%** | **~45%** |

So the majority of miner rewards still come from **UAV**, not from live KAV.

If unused UAV weight has nowhere to go during Sandbox (scores depleted / not yet accumulating), that unused UAV share can be **rerouted to the dual-incentive partner** instead of being burned — which can temporarily make partner / dual-incentive share look larger until Execution restores the normal mix. If the partner hotkey is absent from the metagraph, the 50% partner fraction falls back to the burn UID.

---

## KAV vs UAV

| | **KAV** (Known Attack Vector) | **UAV** (Unknown Attack Vector) |
|---|-------------------------------|----------------------------------|
| **What it is** | Live, per-round image quality score | Reputation from post-validation |
| **When paid** | Continuously on-chain during the cycle | Finalized after review; paid especially in **Execution** |
| **What is graded** | Face image variations (grading API) | Face + screen replay + voice (manual / policy review) |
| **Weight** | 10% of miner share | 90% of miner share |
| **Nature** | Provisional prediction of post-validation | Settled ground truth for reputation |

Live KAV is a **prediction** of what post-validation will confirm. UAV is where scores are finalized and where most rewards land.

### How live KAV works (validator / `reward.py`)

Miners still submit all five face variations, one screen replay, and a voice clip — but **live KAV only grades images**, and only **one** of the five face-variation slots, chosen at random that round.

1. That single sampled variation is graded live via `/grade_v2` (validation score + how well **identity** was preserved).
2. Ranking is driven mainly by that validation score; identity preservation is the tiebreaker and a gate (**≥ 0.6** among the top 50).
3. Qualified miners get a blended exponential-decay reward; everyone else gets **0** KAV for the round.

Screen replay and voice are **not** part of live KAV — they feed UAV post-grading instead.

### How UAV feeds on-chain weights

When UAV grading is enabled, the validator combines:

- `kav_weight × Q` (online quality)
- `uav_weight × (normalized_rep_score × tier_multiplier)`

then rescales so the miner pool hits the configured KAV/UAV targets after partner allocation (and any residual burn). Unqueried miners (not sampled for that round’s KAV challenge) can still receive **UAV-only** rewards from their reputation snapshot; new miners not in the snapshot get **KAV only**.

---

## Cycle structure: Sandbox and Execution

Every cycle has two sections: **Sandbox** and **Execution**. Rewards are always conceptually split into **KAV** and **UAV**, but **what gets paid when** differs by section.

### Sandbox

- Miners **experiment** — try new variation types, voice, liveness attacks, etc., and figure out the problem.
- There is **no UAV** for the current cycle: prediction scores do **not** increase, and no new UAV rewards accumulate or pay out.
- Reviewers use this time to **finalize grades for the previous cycle** (settling that cycle’s UAV).
- Live **KAV** may still run, but Sandbox is practice / calibration — not the UAV payout window.

### Execution

- **Finalized UAV** from the previous cycle is distributed.
- **UAV prediction scores begin accumulating** for the **current** cycle (face + voice review pipeline above).
- Live **KAV** continues as usual.
- Emission mix returns toward the normal **50% miners / 50% dual incentive / 0% burn** picture once UAV is paying again alongside KAV.

### Example (Phase 5, Cycle 1)

| Section | What happens |
|---------|----------------|
| **Sandbox** | Experiment / calibrate; no current-cycle UAV; reviewers finalize **Phase 4 Cycle 5** UAV grades |
| **Execution** (current) | Pay finalized **P4C5 UAV**; accumulate **P5C1** UAV prediction scores (voice 50% + face/screen 50%); KAV continues live |

Sandbox is for figuring things out. Execution is when current-cycle UAV prediction builds and previous-cycle UAV pays.

---

## Quick reference

```
Emissions (weights)
├── ~50% data-generation miners
│     ├── 10% KAV  → ~5% of total  (live image grading each round)
│     └── 90% UAV  → ~45% of total (post-validation reputation; most of miner rewards)
├── ~50% dual-incentive / partner pool (UID 14)
└── ~0% burn (removed)

Point totals
├── Face variations  → out of 25  (5 images × up to 5 points each)
├── Screen replay    → out of 6   (1 image × up to 6 points)
└── Voice            → out of 5   (up to 5 points)

This cycle UAV prediction (P5C1 Execution)
├── 50% face UAV
│     ├── 50% → face variations (out of 25) → apply delta
│     │     └── up to 5 pts/image on variation quality + identity preserved
│     ├── 50% → screen replay (out of 6) → apply delta
│     │     ├── 6 pts if breaks PL v3 + correct cues + correct device + identity
│     │     └── 3 pts if does NOT break PL v3 + correct cues + correct device + identity
│     └── face UAV ← average( face_var , screen )
│           (≈ 25% + 25% of final UAV)
└── 50% voice UAV
      ├── voice (out of 5) → apply delta
      │     ├── 5 pts if tokens match prompt + SpeechBrain identity ≥ 0.6
      │     └── 3 pts if a couple of token mismatches + identity still preserved
      └── voice UAV prediction

final_UAV ≈ average(face_UAV, voice_UAV)
            (* age and other factors)
```

---

## Related docs

- [Miner guide](miner.md) — setup, variation types, voice clone challenge
- [Validator guide](validator.md) — how validators query and set weights
- Miner API: `POST /is_live` (PL v3), `POST /is_ai` (deepfake / face)
- Implementation: `MIID/validator/reward.py` (`get_image_variation_rewards`, `apply_reputation_rewards`)
