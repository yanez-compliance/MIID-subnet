from math import comb

def _kth_composition_lex(n: int, S: int, k_in_S: int) -> tuple[int, ...]:
    """
    k_in_S-th (1-based) n-tuple of nonnegatives summing to S, in lex order.
    """
    total = comb(S + n - 1, n - 1)
    if not (1 <= k_in_S <= total):
        raise IndexError(f"k_in_S out of range: 1..{total}")

    res, rem, k = [], S, k_in_S
    for i in range(n - 1):
        # try c_i = t in [0..rem]; block size for each t:
        #   C(rem - t + (n-i-2), n-i-2)
        for t in range(rem + 1):
            block = comb(rem - t + (n - i - 2), n - i - 2) if n - i - 2 >= 0 else (1 if rem - t == 0 else 0)
            if k <= block:
                res.append(t)
                rem -= t
                break
            k -= block
    res.append(rem)
    return tuple(res)

def kth_plan(n: int, k: int) -> tuple[tuple[int, ...], int]:
    """
    Among ALL n-tuples of nonnegative ints with sum s in {1,2,3,...},
    ordered by increasing s then lex within s,
    return ( (c1,...,cn), s ) for 1-based index k.
    """
    if n <= 0 or k <= 0:
        raise ValueError("n >= 1 and k >= 1 required")

    # cum(S) = total # up to and including sum S (excluding sum 0):
    #   cum(S) = sum_{t=1..S} C(t+n-1, n-1) = C(S+n, n) - 1
    def cum(S: int) -> int:
        return comb(S + n, n) - 1

    # find smallest S with cum(S) >= k  (binary search with exponential grow of hi)
    lo, hi = 1, 1
    while cum(hi) < k:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if cum(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    S = lo

    k_in_S = k - (cum(S - 1) if S > 1 else 0)  # 1-based index within sum=S bucket
    c_tuple = _kth_composition_lex(n, S, k_in_S)
    return c_tuple, S
