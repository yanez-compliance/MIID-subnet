from collections import deque

class MaxFlow:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.to = []
        self.cap = []
        self.rev = []

    def add_edge(self, u, v, c):
        # forward
        self.to.append(v); self.cap.append(c); self.rev.append(len(self.adj[v]))
        self.adj[u].append(len(self.to)-1)
        # backward
        self.to.append(u); self.cap.append(0); self.rev.append(len(self.adj[u])-1)
        self.adj[v].append(len(self.to)-1)

    def maxflow(self, s, t):
        flow = 0
        INF = 10**18
        while True:
            parent_edge = [-1]*self.n
            q = deque([s])
            parent_edge[s] = -2
            bottleneck = [0]*self.n
            bottleneck[s] = INF
            while q and parent_edge[t] == -1:
                u = q.popleft()
                for ei in self.adj[u]:
                    v = self.to[ei]
                    if parent_edge[v] == -1 and self.cap[ei] > 0:
                        parent_edge[v] = ei
                        bottleneck[v] = min(bottleneck[u], self.cap[ei])
                        if v == t: break
                        q.append(v)
            if parent_edge[t] == -1:
                break
            aug = bottleneck[t]
            flow += aug
            v = t
            while v != s:
                ei = parent_edge[v]
                self.cap[ei] -= aug
                # reverse edge index is ei^1 (paired), but we stored explicit rev[]
                # Find reverse quickly:
                u = self.to[ei ^ 1]  # since we always add edges in pairs
                # increase reverse capacity
                self.cap[ei ^ 1] += aug
                v = u
        return flow
def trans_x(x):
    trans_x = [[0]*len(x) for _ in range(len(x[0]))]
    for i in range(len(x)):
        for j in range(len(x[i])):
            trans_x[j][i] = x[i][j]
    return trans_x

# def trans_x(x):
#     trans_x = [[0]*3 for _ in range(4)]
#     for o_level in range(4):
#         for p_level in range(3): 
#             trans_x[p_level][o_level] = x[o_level][p_level]
#     return trans_x

def solve_maxflow(Max, P, O):
    # nodes: s=0, rows 1..3, cols 4..7, t=8
    Max = trans_x(Max)
    
    s, t = 0, 8
    mf = MaxFlow(9)

    # s -> rows
    for i in range(3):
        mf.add_edge(s, 1+i, P[i])

    # rows -> cols, keep indices to read flows later
    rc_edge_idx = [[None]*4 for _ in range(3)]
    for i in range(3):
        for j in range(4):
            u = 1+i
            v = 4+j
            # record the index of the forward edge being added
            forward_index_before = len(mf.to)
            mf.add_edge(u, v, Max[i][j])
            rc_edge_idx[i][j] = forward_index_before  # index of forward edge

    # cols -> t
    for j in range(4):
        mf.add_edge(4+j, t, O[j])

    K_star = mf.maxflow(s, t)

    # Recover x_ij = original_cap - residual_cap on forward edges row->col
    X = [[0]*4 for _ in range(3)]
    for i in range(3):
        for j in range(4):
            ei = rc_edge_idx[i][j]
            # originally we added a forward edge with capacity Max[i][j]
            # After flow, residual cap is mf.cap[ei]
            X[i][j] = Max[i][j] - mf.cap[ei]
    X = trans_x(X)
    return X, K_star


def max_transport33(MaxU, O, P):
    """
    Maximize sum x_ij subject to:
      0 <= x_ij <= MaxU[i][j], sum_j x_ij <= P[i], sum_i x_ij <= O[j]
    Inputs:
      MaxU: 3x3 nonnegative capacities
      O: length-3 column caps
      P: length-3 row caps
    Returns: (x, flow_value)
      x: 3x3 optimal matrix
    """
    n_rows, n_cols = 3, 3
    # Node indexing: s=0, rows=1..3, cols=4..6, t=7
    S, T = 0, 7
    def ridx(i): return 1 + i           # row node
    def cidx(j): return 4 + j           # col node

    N = 8
    cap = [[0]*N for _ in range(N)]
    adj = [[] for _ in range(N)]
    def add(u,v,w):
        if v not in adj[u]: adj[u].append(v)
        if u not in adj[v]: adj[v].append(u)
        cap[u][v] += w  # allow multi-edges merge

    # s -> rows
    for i in range(n_rows):
        add(S, ridx(i), P[i])
    # rows -> cols (store to recover flows)
    for i in range(n_rows):
        for j in range(n_cols):
            add(ridx(i), cidx(j), MaxU[i][j])
    # cols -> t
    for j in range(n_cols):
        add(cidx(j), T, O[j])

    def bfs():
        parent = [-1]*N; parent[S] = S
        q = deque([S])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    if v == T:
                        # reconstruct bottleneck
                        f = float('inf'); x = T
                        while x != S:
                            f = min(f, cap[parent[x]][x])
                            x = parent[x]
                        # apply
                        x = T
                        while x != S:
                            u2 = parent[x]
                            cap[u2][x] -= f
                            cap[x][u2] += f
                            x = u2
                        return f, parent
                    q.append(v)
        return 0, parent

    flow = 0
    while True:
        pushed, _ = bfs()
        if pushed == 0: break
        flow += pushed

    # Recover x: original forward capacity minus residual on row->col arcs
    x = [[0]*n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        u = ridx(i)
        for j in range(n_cols):
            v = cidx(j)
            used = MaxU[i][j] - cap[u][v]
            x[i][j] = used
    return x, flow

def max_transport44(MaxU, O, P):
    """
    Maximize sum x_ij subject to:
      0 <= x_ij <= MaxU[i][j], sum_j x_ij <= P[i], sum_i x_ij <= O[j]
    Inputs:
      MaxU: 4x4 nonnegative capacities
      O: length-4 column caps
      P: length-4 row caps
    Returns: (x, flow_value)
      x: 4x4 optimal matrix
    """
    n_rows, n_cols = 4, 4
    # Node indexing: s=0, rows=1..4, cols=5..8, t=9
    S, T = 0, 9
    def ridx(i): return 1 + i           # row node
    def cidx(j): return 5 + j           # col node

    N = 10
    cap = [[0]*N for _ in range(N)]
    adj = [[] for _ in range(N)]
    def add(u,v,w):
        if v not in adj[u]: adj[u].append(v)
        if u not in adj[v]: adj[v].append(u)
        cap[u][v] += w  # allow multi-edges merge

    # s -> rows
    for i in range(n_rows):
        add(S, ridx(i), P[i])
    # rows -> cols (store to recover flows)
    for i in range(n_rows):
        for j in range(n_cols):
            add(ridx(i), cidx(j), MaxU[i][j])
    # cols -> t
    for j in range(n_cols):
        add(cidx(j), T, O[j])

    def bfs():
        parent = [-1]*N; parent[S] = S
        q = deque([S])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    if v == T:
                        # reconstruct bottleneck
                        f = float('inf'); x = T
                        while x != S:
                            f = min(f, cap[parent[x]][x])
                            x = parent[x]
                        # apply
                        x = T
                        while x != S:
                            u2 = parent[x]
                            cap[u2][x] -= f
                            cap[x][u2] += f
                            x = u2
                        return f, parent
                    q.append(v)
        return 0, parent

    flow = 0
    while True:
        pushed, _ = bfs()
        if pushed == 0: break
        flow += pushed

    # Recover x: original forward capacity minus residual on row->col arcs
    x = [[0]*n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        u = ridx(i)
        for j in range(n_cols):
            v = cidx(j)
            used = MaxU[i][j] - cap[u][v]
            x[i][j] = used
    return x, flow



####################
from heapq import heappush, heappop

class MCMF:
    class Edge:
        __slots__ = ("to","rev","cap","cost","orig")
        def __init__(self, to, rev, cap, cost):
            self.to = to
            self.rev = rev
            self.cap = cap
            self.cost = cost
            self.orig = cap

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        fwd = MCMF.Edge(v, len(self.g[v]), cap, cost)
        rev = MCMF.Edge(u, len(self.g[u]), 0, -cost)
        self.g[u].append(fwd)
        self.g[v].append(rev)
        return (u, len(self.g[u]) - 1)  # handle to forward edge

    def min_cost_flow(self, s, t, need):
        n = self.n
        flow = 0
        cost = 0
        INF = 10**18
        # costs are non-negative; Dijkstra suffices
        while flow < need:
            dist = [INF] * n
            prev = [(-1, -1)] * n
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, u = heappop(pq)
                if d != dist[u]: continue
                for ei, e in enumerate(self.g[u]):
                    if e.cap <= 0: continue
                    v = e.to
                    nd = d + e.cost
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = (u, ei)
                        heappush(pq, (nd, v))
            if dist[t] == INF:
                # cannot push the required amount
                break
            # augment 1 unit (all capacities are integer; we can also push more)
            add = need - flow
            # find bottleneck
            v = t
            while v != s:
                u, ei = prev[v]
                add = min(add, self.g[u][ei].cap)
                v = u
            # apply
            v = t
            while v != s:
                u, ei = prev[v]
                e = self.g[u][ei]
                e.cap -= add
                self.g[v][e.rev].cap += add
                v = u
            flow += add
            cost += add * dist[t]
        return flow, cost

def solve_integer_diverse(Max, O):
    """
    Max: 3x4 matrix of nonnegative integers (upper bounds)
    O  : length-4 list of nonnegative integers (column budgets)
    Returns: x (3x4 integer matrix), row_sums (len 3)
    """
    R, C = 3, 4
    Max = trans_x(Max)
    if len(Max) == 1:
        return trans_x([O]), trans_x([O])

    assert len(Max) == R and all(len(row) == C for row in Max)
    assert len(O) == C

    # Effective per-column supply (force full feasible use)
    t = [min(O[j], sum(Max[i][j] for i in range(R))) for j in range(C)]
    total = sum(t)
    # Row upper bounds
    row_cap = [sum(Max[i][j] for j in range(C)) for i in range(R)]
    # Integer target average
    avg = round(total / R)

    # Build graph
    # nodes: S | C0..C3 | R0..R2 | T
    S = 0
    C0 = 1
    R0 = C0 + C
    T  = R0 + R
    N  = T + 1
    g = MCMF(N)

    # S -> columns
    for j in range(C):
        g.add_edge(S, C0 + j, t[j], 0)

    # columns -> rows (store handles to extract x later)
    rc_handle = [[None]*C for _ in range(R)]
    for j in range(C):
        for i in range(R):
            rc_handle[i][j] = g.add_edge(C0 + j, R0 + i, Max[i][j], 0)

    # rows -> T as unit arcs with convex increasing cost around avg
    # Incremental cost for k-th unit on a row: c(k) = 2*k + 1 - 2*avg
    # Shift all costs to be non-negative (doesn't change argmin since total flow is fixed)
    shift = max(0, 2*avg - 1)
    for i in range(R):
        K = row_cap[i]
        for k in range(K):  # add K parallel unit-cap edges
            inc_cost = (2*k + 1 - 2*avg) + shift
            g.add_edge(R0 + i, T, 1, inc_cost)

    # Push exactly 'total' units
    pushed, _ = g.min_cost_flow(S, T, total)
    if pushed != total:
        raise RuntimeError("Infeasible: not enough capacity to realize all t_j")

    # Extract integer x from column->row edges
    x = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            u, idx = rc_handle[i][j]
            e = g.g[u][idx]
            used = e.orig - e.cap
            x[i][j] = used

    row_sums = [sum(x[i][j] for j in range(C)) for i in range(R)]
    return trans_x(x), row_sums

def solve_integer_diverse_min(Max, O):
    """
    Diverse *minimizing* integer flow (i.e., concentrate mass on as few rows as possible)
    using min-cost max-flow with *decreasing* marginal costs per row.

    Max: 3x4 matrix of nonnegative integers (upper bounds per row/col)
    O  : length-4 list of nonnegative integers (column budgets)
    Returns: x (3x4 integer matrix), row_sums (len 3)
    """
    R, C = 3, 4
    Max = trans_x(Max)
    if len(Max) == 1:
        return trans_x([O]), trans_x([O])

    assert len(Max) == R and all(len(row) == C for row in Max)
    assert len(O) == C

    # Effective per-column supply (cap by what rows can actually accept)
    t = [min(O[j], sum(Max[i][j] for i in range(R))) for j in range(C)]
    total = sum(t)

    # Row capacity (max possible into each row)
    row_cap = [sum(Max[i][j] for j in range(C)) for i in range(R)]

    # For symmetry with the previous version, keep the same avg reference,
    # but we will make the marginal costs *decrease* with k to favor concentration.
    avg = round(total / R)

    # Build graph: S | C0..C3 | R0..R2 | T
    S = 0
    C0 = 1
    R0 = C0 + C
    T  = R0 + R
    Nn = T + 1
    g = MCMF(Nn)

    # S -> columns
    for j in range(C):
        g.add_edge(S, C0 + j, t[j], 0)

    # columns -> rows (store handles to extract x later)
    rc_handle = [[None]*C for _ in range(R)]
    for j in range(C):
        for i in range(R):
            rc_handle[i][j] = g.add_edge(C0 + j, R0 + i, Max[i][j], 0)

    # rows -> T with *decreasing* incremental cost:
    #   For the k-th unit on a row (k=0-based):
    #       inc_cost_dec(k) = (2*avg - (2*k + 1)) + shift_i
    #   => total cost for r units (before shift) = 2*avg*r - r^2 (concave),
    #      so min-cost prefers packing many units into the same row (low diversity).
    for i in range(R):
        K = row_cap[i]
        # Choose a per-row shift so all arc costs are non-negative for k in [0..K-1]
        # Minimum of (2*avg - (2*k + 1)) occurs at k=K-1: value = 2*avg - (2*K - 1)
        shift_i = max(0, (2*K - 1) - 2*avg)
        for k in range(K):
            inc_cost = (2*avg - (2*k + 1)) + shift_i
            g.add_edge(R0 + i, T, 1, inc_cost)

    # Push exactly 'total' units
    pushed, _ = g.min_cost_flow(S, T, total)
    if pushed != total:
        raise RuntimeError("Infeasible: not enough capacity to realize all t_j")

    # Extract integer x from column->row edges
    x = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            u, idx = rc_handle[i][j]
            e = g.g[u][idx]
            used = e.orig - e.cap
            x[i][j] = used

    row_sums = [sum(x[i][j] for j in range(C)) for i in range(R)]
    return trans_x(x), row_sums



from typing import List, Tuple

def maxflow_then_maxdisp_int(MaxU: List[List[int]], O: List[int]) -> Tuple[List[List[int]], int]:
    """
    Lexicographic objective with integer capacities:
      1) maximize sum_{i,j} x_ij
      2) among max-flow solutions, maximize sum_i (S_i - avg)^2,
         where S_i = sum_j x_ij and avg = (sum_i S_i)/rows

    Subject to:
      0 <= x_ij <= MaxU[i][j]  (integers)
      sum_i x_ij <= O[j]       (integers)

    Inputs:
      MaxU: matrix of size 1x4 or 3x4 with non-negative integers
      O:    list of 4 non-negative integers (column caps)

    Returns:
      x: integer matrix same shape as MaxU (optimal under the lexicographic objective)
      flow_value: integer total flow
    """
    # # ---- validate shapes and types ----
    # if not isinstance(MaxU, list) or len(MaxU) not in (1, 3) or any(len(r) != 4 for r in MaxU):
    #     raise ValueError("MaxU must be 1x4 or 3x4.")
    # if not isinstance(O, list) or len(O) != 4:
    #     raise ValueError("O must have length 4.")
    # if any((not isinstance(v, int)) or v < 0 for r in MaxU for v in r):
    #     raise ValueError("All MaxU entries must be non-negative integers.")
    # if any((not isinstance(v, int)) or v < 0 for v in O):
    #     raise ValueError("All O entries must be non-negative integers.")

    MaxU = trans_x(MaxU)
    rows, cols = len(MaxU), 4

    # ---- Step 1: compute per-column max flow (max-flow) ----
    col_from_rows = [sum(MaxU[i][j] for i in range(rows)) for j in range(cols)]
    F = [min(O[j], col_from_rows[j]) for j in range(cols)]  # integer by construction
    flow_value = sum(F)

    # ---- Precompute row absorbable capacity given F (for dispersion tie-break) ----
    # R[i] = sum_j min(MaxU[i][j], F[j])
    R = [sum(min(MaxU[i][j], F[j]) for j in range(cols)) for i in range(rows)]

    # Row order: most absorbable first (ties by row index for determinism)
    row_order = sorted(range(rows), key=lambda i: (-R[i], i))

    # ---- Step 2: allocate each column's F[j] concentrating mass by row_order ----
    x = [[0 for _ in range(cols)] for _ in range(rows)]

    for j in range(cols):
        f = F[j]
        if f == 0:
            continue
        for i in row_order:
            if f == 0:
                break
            give = min(f, MaxU[i][j] - x[i][j])
            if give > 0:
                x[i][j] += give
                f -= give
        # Since F[j] <= sum_i MaxU[i][j], f must be 0 now.

    # ---- sanity checks ----
    # bounds
    for i in range(rows):
        for j in range(cols):
            if not (0 <= x[i][j] <= MaxU[i][j]):
                raise AssertionError("x out of bounds at (%d,%d)" % (i, j))
    # column caps
    for j in range(cols):
        if sum(x[i][j] for i in range(rows)) > O[j]:
            raise AssertionError("column cap violated at j=%d" % j)
    # total flow
    if sum(sum(row) for row in x) != flow_value:
        raise AssertionError("flow value mismatch.")

    return trans_x(x), flow_value

# ---------- examples ----------
if __name__ == "__main__":
    # 3x4 example
    MaxU_3x4 = [
        [3, 2, 5, 4],
        [2, 8, 1, 3],
        [7, 1, 2, 6],
    ]
    O = [7, 6, 5, 8]
    x, val = maxflow_then_maxdisp_int(trans_x(MaxU_3x4), O)
    print("3x4 solution:")
    for row in x: print(row)
    print("flow_value:", val)

    # 1x4 example
    MaxU_1x4 = [[5, 0, 9, 2]]
    O2 = [3, 7, 10, 1]
    x2, val2 = maxflow_then_maxdisp_int(trans_x(MaxU_1x4), O2)
    print("\n1x4 solution:")
    for row in x2: print(row)
    print("flow_value:", val2)




from heapq import heappush, heappop

class MinCostMaxFlow:
    class Edge:
        __slots__ = ("to", "rev", "cap", "cost", "orig")
        def __init__(self, to, rev, cap, cost):
            self.to   = to
            self.rev  = rev
            self.cap  = cap
            self.cost = cost
            self.orig = cap

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        a = MinCostMaxFlow.Edge(v, len(self.g[v]), cap, cost)
        b = MinCostMaxFlow.Edge(u, len(self.g[u]), 0,  -cost)
        self.g[u].append(a)
        self.g[v].append(b)
        return (u, len(self.g[u]) - 1)  # handle to forward edge

    def min_cost_flow(self, s, t, maxf=None):
        n = self.n
        flow, cost = 0, 0
        pot = [0] * n  # potentials (Johnson's trick); costs are non-negative so this stays 0
        INF = 10**18

        while True:
            dist = [INF] * n
            inq  = [False] * n
            par  = [(-1, -1)] * n  # (node, edge_index)

            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, u = heappop(pq)
                if d != dist[u]:
                    continue
                for idx, e in enumerate(self.g[u]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + pot[u] - pot[e.to]
                    if nd < dist[e.to]:
                        dist[e.to] = nd
                        par[e.to]  = (u, idx)
                        heappush(pq, (nd, e.to))

            if dist[t] == INF:
                break

            for v in range(n):
                if dist[v] < INF:
                    pot[v] += dist[v]

            # bottleneck
            add = INF
            v = t
            while v != s:
                u, idx = par[v]
                add = min(add, self.g[u][idx].cap)
                v = u

            if add == 0 or (maxf is not None and flow == maxf):
                break

            if maxf is not None:
                add = min(add, maxf - flow)

            # augment
            v = t
            path_cost = 0
            while v != s:
                u, idx = par[v]
                e = self.g[u][idx]
                path_cost += e.cost
                e.cap -= add
                self.g[v][e.rev].cap += add
                v = u

            flow += add
            cost += add * path_cost

            if maxf is not None and flow == maxf:
                break

            # if no positive-cost edges were used and maxf is None,
            # loop continues until no more augmenting path

        return flow, cost


def solve_lex_maxflow_minvariance(MaxU, O):
    """
    Lexicographic objective with integer capacities:
      1) maximize total flow sum_{i,j} x_ij
      2) among max-flow solutions, minimize sum_i (S_i - avg)^2
         where S_i = sum_j x_ij and avg = (sum_i S_i)/rows

    Inputs:
      MaxU: matrix of size 1x4 or 3x4 (non-negative integers)
      O   : list of 4 non-negative integers (column caps)

    Returns:
      x: integer matrix same shape as MaxU (optimal under the lexicographic objective)
      flow_value: integer total flow
    """
    # --- normalize inputs ---
    MaxU = trans_x(MaxU)
    if not (isinstance(MaxU, list) and all(isinstance(row, list) for row in MaxU)):
        raise ValueError("MaxU must be a list of lists.")
    if len(O) != 4:
        raise ValueError("O must be length 4.")

    R = len(MaxU)
    if R not in (1, 3):
        raise ValueError("MaxU must have 1 or 3 rows.")
    C = 4
    for r in range(R):
        if len(MaxU[r]) != C:
            raise ValueError("Each row of MaxU must have length 4.")
        if any(x < 0 for x in MaxU[r]):
            raise ValueError("MaxU entries must be non-negative integers.")
    if any(v < 0 for v in O):
        raise ValueError("O entries must be non-negative integers.")

    # Trivial 1x4 case: only one row, max flow is simple min per column
    if R == 1:
        x_row = [min(MaxU[0][j], O[j]) for j in range(C)]
        return trans_x([x_row]), sum(x_row)

    # Effective per-column supply (cannot exceed what rows can absorb)
    col_caps = [min(O[j], sum(MaxU[i][j] for i in range(R))) for j in range(C)]
    total_possible = sum(col_caps)
    row_caps = [sum(MaxU[i][j] for j in range(C)) for i in range(R)]

    # Node layout: S | C0..C3 | R0..R{R-1} | T
    S = 0
    C0 = 1
    R0 = C0 + C
    T  = R0 + R
    Nn = T + 1

    # ---------- Stage 1: find maximum flow (costs = 0) ----------
    g1 = MinCostMaxFlow(Nn)

    # S -> columns
    for j in range(C):
        g1.add_edge(S, C0 + j, col_caps[j], 0)

    # columns -> rows
    for j in range(C):
        for i in range(R):
            cap = MaxU[i][j]
            if cap > 0:
                g1.add_edge(C0 + j, R0 + i, cap, 0)

    # rows -> T (enough capacity)
    for i in range(R):
        if row_caps[i] > 0:
            g1.add_edge(R0 + i, T, row_caps[i], 0)

    F_star, _ = g1.min_cost_flow(S, T, maxf=None)  # maximum flow value

    if F_star == 0:
        # No feasible flow
        return trans_x([[0]*C for _ in range(R)]), 0

    # ---------- Stage 2: among max flows, minimize sum_i S_i^2 ----------
    # Note: sum_i (S_i - avg)^2 = sum_i S_i^2 - const, since avg = F_star / R and sum_i S_i = F_star
    # So minimizing variance is equivalent to minimizing sum_i S_i^2.
    # We realize S_i^2 via unit arcs with increasing marginal costs: k-th unit to row i costs (2k+1).

    g2 = MinCostMaxFlow(Nn)
    # Keep handles to column->row edges to read back flows later
    handles = [[None]*C for _ in range(R)]

    # S -> columns
    for j in range(C):
        g2.add_edge(S, C0 + j, col_caps[j], 0)

    # columns -> rows (capacity as given, zero cost)
    for j in range(C):
        for i in range(R):
            cap = MaxU[i][j]
            if cap > 0:
                handles[i][j] = g2.add_edge(C0 + j, R0 + i, cap, 0)
            else:
                handles[i][j] = None

    # rows -> T : unit-capacity arcs with costs 2k+1 (k = 0..row_caps[i]-1)
    for i in range(R):
        K = row_caps[i]
        for k in range(K):
            g2.add_edge(R0 + i, T, 1, 2*k + 1)

    # push exactly F_star units
    flowed, _ = g2.min_cost_flow(S, T, maxf=F_star)
    assert flowed == F_star, "Internal error: cannot realize max flow in stage 2."

    # extract x from column->row edges
    x = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            h = handles[i][j]
            if h is None:
                x[i][j] = 0
            else:
                u, idx = h
                e = g2.g[u][idx]
                used = e.orig - e.cap  # how much flowed on this edge
                x[i][j] = used

    return trans_x(x), F_star

def _raise_to_level(base, caps, s, iters=60):
    """
    Solve: y_i = clip(L - base_i, 0, caps_i) with sum(y) = s.
    Returns y (list).
    """
    base = [float(b) for b in base]
    caps = [float(c) for c in caps]
    lo = min(base)
    hi = max(b + c for b, c in zip(base, caps))
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        total = 0.0
        for b, c in zip(base, caps):
            total += max(0.0, min(c, mid - b))
        if total < s:
            lo = mid
        else:
            hi = mid
    L = (lo + hi) / 2.0
    y = [max(0.0, min(c, L - b)) for b, c in zip(base, caps)]
    # tiny numerical fix to hit s exactly
    diff = s - sum(y)
    if abs(diff) > 1e-9:
        for i in range(len(y)):
            room_up = caps[i] - y[i]
            room_dn = y[i]
            if diff > 0 and room_up > 0:
                add = min(diff, room_up)
                y[i] += add
                diff -= add
            elif diff < 0 and room_dn > 0:
                take = min(-diff, room_dn)
                y[i] -= take
                diff += take
            if abs(diff) <= 1e-12:
                break
    return y

def solve_min_row_variance(Max, O, max_sweeps=50, tol=1e-9):
    """
    Max: 3x4 nonnegative matrix
    O:   length-4 nonnegative list (column budgets)

    Returns:
      x: 3x4 optimal matrix
      r: length-3 row sums
      avg, obj: target average and objective value
    """
    Max = trans_x(Max)
    nrows, ncols = len(Max), 4
    assert len(Max) == nrows and all(len(row) == ncols for row in Max)
    assert len(O) == ncols

    # Column targets: use as much as feasible
    s = [min(float(O[j]), sum(Max[i][j] for i in range(nrows))) for j in range(ncols)]

    # Initialize
    x = [[0.0]*ncols for _ in range(nrows)]
    r = [0.0]*nrows
    avg = sum(O) / nrows

    # Cyclic coordinate descent over columns
    for _ in range(max_sweeps):
        delta = 0.0
        for j in range(ncols):
            caps = [Max[i][j] for i in range(nrows)]
            # remove current column from row sums to get base
            base = [r[i] - x[i][j] for i in range(nrows)]
            y_new = _raise_to_level(base, caps, s[j])
            # measure change
            for i in range(nrows):
                delta = max(delta, abs(y_new[i] - x[i][j]))
            # commit
            for i in range(nrows):
                r[i] = base[i] + y_new[i]
                x[i][j] = round(y_new[i])
        if delta <= tol:
            break

    obj = sum((ri - avg)**2 for ri in r)
    return trans_x(x), r #, avg, obj

