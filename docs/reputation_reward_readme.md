# YANEZ MIID — Reputation-Weighted Reward System (Phase 3)

This document describes the **big-picture architecture**, **data flows**, **math**, and **step‑by‑step implementation tasks** required to integrate the new **Reputation‑Weighted Reward System** between Validators and the Flask App (Core Machine).

It includes:
- High‑level system overview (full diagram logic in text)
- Reputation snapshot lifecycle
- Reward allocation lifecycle
- Math used for combining reputation with validator online quality scores
- Fully structured task list you can paste into your code agent
- API schemas and code‑level expectations

---

# 1. BIG PICTURE OVERVIEW

Phase 3 of the YANEZ MIID subnet introduces **Reputation‑Based Rewards**. That means:

### ✔ Validators still compute online quality scores (Phase‑2 logic)
### ✔ But now they request **Reputation Scores** from the Flask API
### ✔ Validators apply multipliers based on `rep_score` and `rep_tier`
### ✔ Validators submit final weighted rewards back to Flask evey hour
### ✔ Flask stores them (JSON for now) → later inserted into DB every hour
### ✔ Reputation for each cycle is computed based on **manual validation** of UAVs

---

# 2. HIGH-LEVEL DATA FLOW

Below is the conceptual flow extracted from the system diagram.

## **A. Manual Validation → DB → Reputation Engine**
1. Human reviewers validate UAVs.
2. Records updated in DB: `validation_score`, `comment`, `status=reviewed`.
3. Reputation Engine Job periodically:
   - Selects `validation_status='reviewed' AND rep_processed_at IS NULL`
   - Computes Δrep per miner
   - Writes rows into `MINER_REPUTATION_HISTORY`
   - Updates `MINER_REPUTATION`
   - Marks UAVs as processed
4. A reputation snapshot is generated hourly/daily.

## **B. Reputation Snapshot → Flask → Validators**
1. Flask caches snapshot in memory as:
   ```json
   {
     "version": "2025‑11‑20T13:00Z",
     "generated_at": "2025‑11‑20T13:00Z",
     "miners": {
       "hotkey1": { "rep_score": 1.23, "rep_tier": "Bronze" }
     }
   }
   ```
2. Validator requests reputation using `/reputation_request`.
3. Flask returns rep data + snapshot version.

## **C. Validator Computes Final Rewards → Sends Back `/reward_allocation`**
1. Validator computes online scores.
2. Applies rep multipliers.
3. Sends final JSON back to Flask.
4. Flask stores JSON (later DB insert).
5. update cycle ends evey hour.

---

# 3. MATH: REPUTATION-WEIGHTED REWARD FORMULA

Online Quality Score (Phase‑2) → `Q`.
Reputation Score → `R` (baseline 1.0).
Reputation Tier Multiplier → `T`.

### **Step 1 — Tier Multiplier**
From config:
```
Diamond: 1.15
Gold:    1.10
Silver:  1.05
Bronze:  1.02
Neutral: 1.00
Watch:   0.90
```

### **Step 2 — Reputation Score Adjustment**
We convert `rep_score` into a second multiplier.

A flexible version:
```
score_factor = 1 + 0.1 * log10(rep_score)
```
Clamp inside safe range:
```
score_factor = clamp(score_factor, 0.8, 1.5)
```

### **Step 3 — Final Reward**
```
final_reward = Q * T * score_factor
```

### **Step 4 — Reputation Bonus (for logging)**
```
reputation_bonus = final_reward - Q
```

---

# 4. API SCHEMAS

## **4.1 `/reputation_request` (Validator → Flask)**
**Request:**
```json
{
  "validator_hotkey": "xxx",
  "miners": ["hotkey1", "hotkey2"],
  "signature": "signed_payload"
}
```

**Response:**
```json
{
  "rep_snapshot_version": "2025‑11‑20T13:00Z",
  "generated_at": "2025‑11‑20T13:00Z",
  "miners": [
    {"miner_hotkey":"hotkey1", "rep_score":1.23, "rep_tier":"Bronze"},
    {"miner_hotkey":"hotkey2", "rep_score":0.95, "rep_tier":"Neutral"}
  ]
}
```

---

## **4.2 `/reward_allocation` (Validator → Flask)**
**Request:**
```json
{
  "validator_hotkey": "xxx",
  "rep_snapshot_version": "2025‑11‑20T13:00Z",
  "cycle_id": "C1",
  "step_id": "block_12345",
  "miners": [
    {
      "miner_hotkey": "hotkey1",
      "base_reward": 0.73,
      "reputation_bonus": 0.12,
      "total_reward": 0.85
    }
  ],
  "signature": "signed_payload"
}
```

**Response:**
```json
{"status": "ok"}
```

---

# 5. WHAT YOU CAN IMPLEMENT NOW (BEFORE DB CONNECTION)

These tasks require **no DB**, only Flask + Validator code.

Below is the clean task list for your code agent.

---

# 6. IMPLEMENTATION TASKS (COPY FOR YOUR CODE AGENT)

## **TASK 1 — Create Reputation Snapshot Models and In-Memory Cache**
- Add `ReputationEntry` dataclass
- Add `ReputationSnapshot` structure
- Add `CURRENT_REP_SNAPSHOT` global
- Add loader from `data/reputation_snapshot.json`
- Auto-load snapshot on Flask startup

---

## **TASK 2 — Implement `/reputation_request` Endpoint**
- Verify validator signature (existing code)
- Parse miner hotkeys
- Lookup in `CURRENT_REP_SNAPSHOT`
- Default missing miners → `{rep_score:1.0, rep_tier:'Neutral'}`
- Return full JSON payload

---

## **TASK 3 — Implement `/reward_allocation` Endpoint**
- Verify validator signature
- Validate JSON
- Save JSON under:
  ```
  data/rewards/<cycle>/<validator>/reward_<timestamp>.json
  ```
- Respond `{ "status": "ok" }`

---

## **TASK 4 — Add Reputation Weighting Logic (`reward.py`)**
Add:
```python
def apply_reputation_weighting(base_reward, rep_score, rep_tier):
    # compute tier_factor
    # compute score_factor via log10
    # clamp
    return base_reward * tier_factor * score_factor
```

---

## **TASK 5 — Integrate Reputation Into Validator Loop (`forward.py`)**
1. Fetch rep snapshot using `/reputation_request`
2. Apply weighting per miner
3. Build reward_allocation payload
4. Sign & POST to Flask

---

## **TASK 6 — Add Flask Client Helpers**
- `fetch_reputation(hotkeys)`
- `send_reward_allocation(payload)`

Use existing `sign_message.py` for signatures.

---

## **TASK 7 — Script to Generate Mock Snapshots**
`generate_mock_snapshot.py`:
- Load list of miners
- Randomize rep_score
- Compute tiers
- Save JSON snapshot

---

# 7. CODE SNIPPETS (READY TO COPY)

## **Flask — Global Snapshot Cache**
```python
CURRENT_REP_SNAPSHOT = None

class ReputationEntry(BaseModel):
    miner_hotkey: str
    rep_score: float
    rep_tier: str

class ReputationSnapshot(BaseModel):
    version: str
    generated_at: str
    miners: dict[str, ReputationEntry]
```

## **Flask — `/reputation_request`**
```python
@app.post("/reputation_request")
def reputation_request():
    body = request.get_json()
    validator_hotkey = body["validator_hotkey"]
    miners = body["miners"]
    signature = body["signature"]

    verify_signature_or_abort(body, signature, validator_hotkey)

    results = []
    for hk in miners:
        entry = CURRENT_REP_SNAPSHOT.miners.get(hk, None)
        if entry is None:
            entry = ReputationEntry(miner_hotkey=hk, rep_score=1.0, rep_tier="Neutral")
        results.append(entry.dict())

    return jsonify({
        "rep_snapshot_version": CURRENT_REP_SNAPSHOT.version,
        "generated_at": CURRENT_REP_SNAPSHOT.generated_at,
        "miners": results
    })
```

## **Validator — Applying Reputation Weighting**
```python
for m in miners:
    rep = rep_data.get(m.hotkey, {"rep_score":1.0, "rep_tier":"Neutral"})
    m.reward_with_rep = apply_reputation_weighting(m.final_reward,
                                                   rep["rep_score"],
                                                   rep["rep_tier"])
```

## **Validator — Sending Final Rewards**
```python
payload = {
  "validator_hotkey": VALIDATOR_HOTKEY,
  "rep_snapshot_version": rep_snapshot_version,
  "cycle_id": cycle_id,
  "step_id": step_id,
  "miners": miners_payload
}

signed = sign_message(payload)
send_reward_allocation(signed)
```

---

# 8. SUMMARY

This README gives you:
- Big picture explanation
- Reputation and reward pipelines
- Mathematical approach
- Exact task breakdown for implementation
- Schema + code ready for your agent

When you are ready, I can also generate:
- The full folder structure
- Complete boilerplate code files
- Postman collection for all endpoints
- DB migration scripts for when the DB connects

Just tell me. 🚀

