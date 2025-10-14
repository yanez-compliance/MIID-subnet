
from math import exp

def optimal_n0_n1_n2(N, targets, boundaries=None, penalty_weight=0.1):
    """
    Return integers (n0, n1, n2) that maximize expected phonetic_quality for total N,
    where:
      n0 = total in c0 (3/3 match)
      n1 = total in c1..c3 combined (2/3 match)
      n2 = total in c4..c6 combined (1/3 match)

    targets: dict like {"Light": tL, "Medium": tM, "Far": tF} (fractions summing to <= 1)
    boundaries: dict with band edges, default: Light=(0.80,1.00), Medium=(0.60,0.79), Far=(0.30,0.59)
    penalty_weight: penalty applied to expected unmatched fraction
    """

    # ---- defaults ----
    if boundaries is None:
        boundaries = {"Light": (0.80, 1.00), "Medium": (0.60, 0.79), "Far": (0.30, 0.59)}
    tL = float(targets.get("Light", 0.0))
    tM = float(targets.get("Medium", 0.0))
    tF = float(targets.get("Far", 0.0))

    # ---- Beta CDFs for Dirichlet(1,1,1) component sums ----
    def beta21_cdf(x: float) -> float:  # Beta(2,1): F(x)=x^2 on [0,1]
        if x <= 0.0: return 0.0
        if x >= 1.0: return 1.0
        return x * x

    def beta12_cdf(x: float) -> float:  # Beta(1,2): F(x)=2x - x^2 on [0,1]
        if x <= 0.0: return 0.0
        if x >= 1.0: return 1.0
        return 2.0 * x - x * x

    def prob_band_beta21(a, b):  # P(a<=X<=b), X~Beta(2,1)
        return max(0.0, min(1.0, beta21_cdf(b) - beta21_cdf(a)))

    def prob_band_beta12(a, b):  # P(a<=X<=b), X~Beta(1,2)
        return max(0.0, min(1.0, beta12_cdf(b) - beta12_cdf(a)))

    # ---- band probabilities for each category type ----
    Llo, Lhi = boundaries["Light"]
    Mlo, Mhi = boundaries["Medium"]
    Flo, Fhi = boundaries["Far"]

    # c0 (3/3 match): score=1.0 → always Light (assuming 1.0 within Light band)
    P0 = {"L": 1.0, "M": 0.0, "F": 0.0, "U": 0.0}

    # c1..c3 (2/3 match): Beta(2,1)
    P2 = {
        "L": prob_band_beta21(Llo, Lhi),
        "M": prob_band_beta21(Mlo, Mhi),
        "F": prob_band_beta21(Flo, Fhi),
    }
    P2["U"] = max(0.0, 1.0 - (P2["L"] + P2["M"] + P2["F"]))

    # c4..c6 (1/3 match): Beta(1,2)
    P1 = {
        "L": prob_band_beta12(Llo, Lhi),
        "M": prob_band_beta12(Mlo, Mhi),
        "F": prob_band_beta12(Flo, Fhi),
    }
    P1["U"] = max(0.0, 1.0 - (P1["L"] + P1["M"] + P1["F"]))

    # ---- scoring helpers ----
    def match_quality(count: float, target_count: float) -> float:
        if target_count <= 0.0:
            return 0.0
        r = count / target_count
        return r if r <= 1.0 else 1.0 - exp(-(r - 1.0))

    def expected_counts(n0: int, n1: int, n2: int):
        L = P0["L"] * n0 + P2["L"] * n1 + P1["L"] * n2
        M = P0["M"] * n0 + P2["M"] * n1 + P1["M"] * n2
        F = P0["F"] * n0 + P2["F"] * n1 + P1["F"] * n2
        U = P0["U"] * n0 + P2["U"] * n1 + P1["U"] * n2
        return L, M, F, U

    def expected_quality(n0: int, n1: int, n2: int):
        L, M, F, U = expected_counts(n0, n1, n2)
        q = 0.0
        q += tL * match_quality(L, tL * N)
        q += tM * match_quality(M, tM * N)
        q += tF * match_quality(F, tF * N)
        q -= penalty_weight * (U / N)
        return q, U

    # ---- exhaustive search over integer triples n0+n1+n2=N ----
    best = None
    best_triplet = (0, 0, 0)
    for n0 in range(N + 1):
        for n1 in range(N - n0 + 1):
            n2 = N - n0 - n1
            q, U = expected_quality(n0, n1, n2)
            # tie-breakers: higher q, then lower unmatched, then smaller n0, then larger n1
            key = (q, -U, -n0, n1)
            if best is None or key > best:
                best = key
                best_triplet = (n0, n1, n2)

    return best_triplet
