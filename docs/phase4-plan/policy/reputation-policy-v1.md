# Reputation Policy V1 (Internal Spec – Subnet 54)

_Last updated: 2025-11-21  
_Status: Internal Only – Defines how miner reputation is computed from manual UAV validation._

## 1. Purpose & Scope

This document defines how **miner reputation** is computed and updated in Subnet 54 based on **Cycle manual validation** of UAVs (Unknown Attack Vectors). Reputation acts as a **long-term trust metric**, separate from per-step emissions, and will be used by validators in the next cycle to weight rewards and track historical performance.

This specification is internal and not intended for public release.

## 2. Core Concepts

### Miner  
A unique participant identified by hotkey, modeled in the `MINER` table.

### UAV (Unknown Attack Vector)  
A miner-submitted data point stored in the UAV table. Relevant fields for reputation include:
- `miner_id`, `cycle_id`
- `validation_status`
- `validation_score`
- `validated_at`
- `is_duplicate`
- `reviewer_id`, `comment`

### AUTO-Validation Score (`validation_score`)  
AUTO-scored integer from **–5 → +5**:
| Score | Meaning |
|-------|---------|
| **+2** | Simple word changes, one to two words are different, missing or added.(target it so catch any spelling changes)(Target is also to catch any simple additions or removal from the address) |
| **0** | UAV is not a UAV, when sent to the api the coordinates match the coordinates sent |
| **0** | +0	Coordinates are not in the same region|
| **–3** | Duplicate – recycled or structurally reused |
| **–5** | UAV is just spam text, simple check to make sure there are more letters then numbers and its not an empty address |


### Validation Score (`validation_score`)  
Human-scored integer from **–5 → +5**:

| Score | Meaning |
|-------|---------|
| **+5** | Perfect – high-quality UAV, realistic, well-labeled |
| **+3/+4** | Good – usable with minor issues |
| **+1/+2** | Acceptable – partial, noisy but still useful |
| **0** | Unclear – insufficient signal |
| **–3** | Duplicate – recycled or structurally reused |
| **–5** | Cheat / fake / exploit |

### Reputation Score (`rep_score`)
A scalar trust metric per miner.

- **Baseline for new miners:** `1.00` (Neutral).  
- **Lower bound (penalty floor):** `0.10`  
- **Reputation never goes negative.**  
- **Miners who repeatedly cheat can be driven toward 0.10.**  
- **No practical upper bound** → for safety our upper bound is `9999.0` to avoid overflows.

### Cycle  
Logical time window such as `C1-Exec`. Reputation updates happen once per batch or cycle, never per individual step.

### Tier (`rep_tier`)  
A coarse label used for dashboards, analytics, and optional weighting logic.

### Miner History and Maturity

Miner reputation is interpreted together with their history and maturity:
- **History**: how many UAVs have been manually validated for this miner (total_uavs_validated), how many cycles they have participated in, and their past reputation changes as recorded in MINER_REPUTATION_HISTORY.
- **Age / Maturity**: how long the miner has been active (time since first validated UAV) and the volume of validated UAVs they have contributed.

These signals are used to distinguish between new, unproven miners and long-term contributors with a stable track record.

## 3. Data Model

### 3.1 UAV Table (inputs to reputation)
Fields used directly:
- `id`
- `miner_id`
- `cycle_id`
- `validation_score`
- `validation_status` (`pending`, `needs_review`, `reviewed`)
- `validated_at`
- `is_duplicate`
- `comment`
- `reviewer_id`
- `rep_processed_at` (timestamp, used for idempotence)

### 3.2 MINER_REPUTATION (current state)
- `miner_id` (PK)
- `rep_score`
- `rep_tier`
- `total_uavs_validated`
- `total_duplicates`
- `total_cheats`
- `last_delta`
- `last_reason`
- `last_updated_at`
- `created_at`

### 3.3 MINER_REPUTATION_HISTORY (audit log)
Each change produces a row:
- `id`
- `miner_id`
- `cycle_id`
- `uav_id` (nullable)
- `delta`
- `reason`
- `rep_score_before`
- `rep_score_after`
- `auto_source`
- `created_at`

## 4. Reputation Scoring Rules (UAV → Δrep)

### 4.1 Inputs per UAV
A UAV contributes to reputation only if:

```sql
validation_status = 'reviewed'
AND rep_processed_at IS NULL
```

### 4.2 Mapping Table (validation_score → Δrep)

| validation_score | Meaning | Δrep |
|------------------|---------|------|
| **+5** | Perfect | +1 |
| **+3/+4** | Good | +0.4 |
| **+1/+2** | Acceptable | +0.2 |
| **0** | Unclear | 0.00 |
| **–3** | Duplicate | –0.10 |
| **–5** | Cheat / fake | –0.50 |

**Penalty > reward** by design.


### 4.4 update rep_score

```
rep_score_after = rep_score = clamp(rep_score_before + delta, 0.10, 9999.0)
```

- **Lower bound (penalty floor): 0.10**  
- **Reputation never goes negative**  
- **Baseline for new miners: 1.00 (Neutral)**  
- **Upper bound** (`9999.0`) is technical only — not a practical limit.

## 5. Tiers

Tiers provide a high-level trust classification for analytics and reward multipliers.

### 5.1 Tier Boundaries (V1)

| Tier | rep_score Range |
|------|-----------------|
| **Diamond** | ≥ 50.0 |
| **Gold** | 30.0 – 49.999 |
| **Silver** | 15.0 – 29.999 |
| **Bronze** | >5.0 – 14.999 |
| **Neutral** | 1 – 5.00 |
| **Watch** | 0.10 – 1 |

### Notes
- **Neutral is exactly baseline (1.0)**  
- **Bronze is strictly above Neutral (rep_score > 1.0)**  
- **Tiers are recomputed whenever rep_score changes**  
- **Miners who repeatedly cheat drift toward 0.10 (floor)**  
- **No practical upper bound: rep_score can grow over many cycles**

### 5.2 Maturity and History Signals

Reputation tiers describe current trust level, but we also consider how much history a miner has behind that score:

- **New miners** – few or zero reviewed UAVs; reputation is close to the baseline (around 1.0) and history is short.
- **Established miners** – a meaningful number of reviewed UAVs (e.g., tens or hundreds) across one or more cycles.
- **Veteran miners** – large history of reviewed UAVs across many cycles, with a stable or improving reputation trajectory.

These categories are not enforced directly in the reputation score, but they are useful for downstream logic (e.g., rewards, dashboards) to avoid treating a brand‑new miner with rep_score = 1.0 the same as a long‑term miner who has maintained rep_score ≈ 1.0 over hundreds of validated UAVs.

## 6. Update Lifecycle

### 6.1 Selecting UAVs for a Run

The reputation job selects UAVs where:

```sql
validation_status = 'reviewed'
AND rep_processed_at IS NULL
```

This guarantees idempotence.

### 6.2 Applying Updates (per miner)

For each miner with at least one selected UAV:

1. **Compute `delta_total`**  
   Sum all Δrep across selected UAVs.

2. **Load or initialize rep_score_before**  
   - If miner has no existing row →  
     `rep_score_before = 1.00` (Neutral)

3. **Compute bounded rep_score_after**  
   Using global floor and ceiling.

4. **Insert MINER_REPUTATION_HISTORY row**
   - `miner_id`, `cycle_id`
   - `delta = delta_total`
   - `rep_score_before`
   - `rep_score_after`
   - `reason = 'validated_uav_batch'`
   - `auto_source = 'manual_validation_job'`
   - `created_at = now()`

5. **Upsert MINER_REPUTATION**
   - Update `rep_score = rep_score_after`
   - Update `rep_tier`
   - Update `last_delta = delta_total`, `last_reason`, `last_updated_at`
   - Increment:
     - `total_uavs_validated`
     - `total_duplicates`
     - `total_cheats`

Finally:

```sql
UPDATE uav SET rep_processed_at = now()
WHERE uav_id IN (all processed UAVs)
```

### 6.3 Idempotence Guarantee

The job does **not** reprocess UAVs because only rows with:

```sql
rep_processed_at IS NULL
```

are selected.

## 7. Edge Cases & Overrides

### New Miner (no reviewed UAVs yet)
If a miner has not yet produced a reviewed UAV:
- No MINER_REPUTATION row may exist.
- When they first enter:
  - `rep_score = 1.0`
  - `rep_tier = Neutral`

### Miner with no new reviewed UAVs in a run
- No changes are made.
- No history row is created.

### Re-Review of a UAV
If a reviewer updates the validation_score:
- Reviewer must update `validated_at`.
- Reviewer must clear `rep_processed_at = NULL`.
- Reputation job will reprocess the UAV.
- Old history rows remain (accepted minor drift).

### Manual Overrides
Operators may adjust a miner's reputation manually.

Required steps:
- Insert a MINER_REPUTATION_HISTORY row with:
  - `reason = 'manual_override'`
  - appropriate `delta`
  - `auto_source = 'ops'`
- Update MINER_REPUTATION to keep scores consistent.

## 8. Interaction With Rewards (Informational)


Validators may use `rep_score` or tier to influence reward distribution.

**Important:** Reputation computed during Cycle 1 (C1) is *not* used for reward distribution within Cycle 1. Reputation collected in C1 will first influence rewards in the subsequent cycle (Cycle 2).

This policy **defines what rep_score means**, but does **not** define reward formulas.  
Reward logic is specified separately.

In practice, validators and reward logic may combine **reputation** and **maturity** when deciding bonuses or weighting. For example:

- A new miner with `rep_score = 1.0` and `total_uavs_validated = 0` may receive **no reputation-based bonus** until they accumulate a minimum amount of validated work.
- A miner with `rep_score = 0.8` but `total_uavs_validated ≈ 200` may still be **fully eligible** for certain reward schemes because they have a long, analyzable history.

These maturity-based adjustments begin to influence miner weighting and bonuses starting in **Cycle 2**, when we begin calculating and updating miner reputation scores and tiers for the new cycle.


These examples are illustrative only; exact thresholds and multipliers live in the validator/reward specification, not in this policy. The key idea is that resetting to a new identity discards historical maturity, while staying and improving preserves it.

## 9. Versioning

### Version Rules
Changes to:
- Δrep mapping  
- Global bounds  
- Tier definitions  
- Idempotence rules  

must be versioned as new policy versions (e.g., V2, V3).

### Changelog
- **V1 (2025-11-XX)** — Initial internal specification. No cycle caps, new tier system, explicit penalty floor, rep_processed_at idempotence.
