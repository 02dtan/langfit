"""Energy-weighted Laplacian-eigenmap evidence (all four models x ell x denominator).

Generalised problem XLX v = mu B v in the indicator space, with
  XLX = sum_{edges} exp(-(dE/ell)^2) (x_s - x_t)(x_s - x_t)^T
and two denominators: B_deg = X^T D_deg X (D_deg = graph degree, uniform L(A-1) on the
full Hamming graph) and B_P = X^T D_P X (D_P = diag(P), the P-weighted second moment).
The two smallest non-trivial generalised eigenvectors give the 2-D layout.  Reports the
coverage, localisation (IPR), eigen-gap, energy-ramp R2, linearity, roughness, and the
standard collapse-guard metrics for each case.  Writes nothing; prints a table.
"""
import numpy as np
import scipy.linalg
from scipy.stats import spearmanr
import langfit.maxent as me
import langfit.projection as pj
from langfit import basis as bs

L, A = 4, 26
G = 32


def load(key):
    npz = np.load("data/derived/landscape_artifact.npz")
    V = npz["V_traj"][-1] if key == "pooled" else npz[f"V_{key}"]
    m = me.MaxEnt(L=L, cliques=me.cliques(2), V=[v.copy() for v in V])
    npz.close()
    P = m.prob()
    lp = m.log_prob().ravel() / np.log(2)
    with np.errstate(divide="ignore"):
        E = (-np.log(P) / np.log(2)).ravel()
    E = bs._cap_infinite(E)
    return P, lp, E


def uniform_second_moment(L, A):
    d = L * A
    S = np.full((d, d), 1.0 / A ** 2)
    for i in range(L):
        S[i * A:(i + 1) * A, i * A:(i + 1) * A] = np.eye(A) / A
    return S


def laplacian_eigenmap(P, E, ell, denom):
    d = L * A
    XLX = np.zeros((d, d))
    Ep_all = E.reshape((A,) * L)
    for p in range(L):
        Ep = np.moveaxis(Ep_all, p, 0).reshape(A, -1)
        for a in range(A):
            for b in range(a + 1, A):
                c = float(np.exp(-((Ep[a] - Ep[b]) ** 2) / ell ** 2).sum())
                ia, ib = p * A + a, p * A + b
                XLX[ia, ia] += c
                XLX[ib, ib] += c
                XLX[ia, ib] -= c
                XLX[ib, ia] -= c
    if denom == "degree":
        B = (L * (A - 1)) * uniform_second_moment(L, A)
    else:  # P-weighted second moment = C0 + mu mu^T
        C0 = bs._indicator_covariance(P)
        mu = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                             for j in range(L)])
        B = C0 + np.outer(mu, mu)
    B = B + 1e-8 * np.trace(B) * np.eye(d)
    lam, V = scipy.linalg.eigh(XLX, B)
    null = int(np.sum(lam < 1e-8 * max(1.0, lam[-1])))
    gap = float((lam[null + 2] - lam[null + 1]) / lam[null + 1]) if null + 2 < d else float("nan")
    U = V[:, null:null + 2]
    U = bs._sign_fix(U / np.linalg.norm(U, axis=0))
    return U, lam[null:null + 3], gap


def metrics(P, lp, E, coords):
    sup = P.ravel() > 0
    bins = pj.bin_index(coords, pj.grid_edges(coords, G))
    Ebin, count = pj.bin_energy(lp, bins, G * G)
    occ = np.isfinite(Ebin)
    Pcell = np.bincount(bins, weights=np.exp2(lp), minlength=G * G)
    # occupancy / collapse guards
    occ_frac = float(occ.sum() / (G * G))
    mass = Pcell[occ]
    pm = mass / mass.sum()
    eff = float(np.exp(-(pm * np.log(pm)).sum()))
    x, y = coords[0], coords[1]
    w = P.ravel()
    # P-weighted axis variance ratio
    xy = coords[:, sup].T
    mu0 = (w[sup][:, None] * xy).sum(0) / w[sup].sum()
    vx = (w[sup] * (xy[:, 0] - mu0[0]) ** 2).sum() / w[sup].sum()
    vy = (w[sup] * (xy[:, 1] - mu0[1]) ** 2).sum() / w[sup].sum()
    varR = max(vx, vy) / min(vx, vy) if min(vx, vy) > 0 else float("inf")
    # roughness (robust SD-normalised)
    sc = bs.mass_weighted_scale(Ebin, Pcell, occ)
    dE = []
    for k in np.flatnonzero(occ):
        r, c = divmod(int(k), G)
        for dr, dc in ((1, 0), (0, 1)):
            if 0 <= r + dr < G and 0 <= c + dc < G:
                j = (r + dr) * G + c + dc
                if occ[j]:
                    dE.append(abs(Ebin[k] - Ebin[j]))
    dE = np.array(dE)
    rough = float(dE.mean() / sc["sd"]) if sc["sd"] > 0 and len(dE) else float("nan")
    # within-cell R2 (P-weighted)
    Efin = -lp
    Efin = np.where(np.isfinite(Efin), Efin, 0.0)
    meanE = float((w * Efin).sum())
    sst = (w * (Efin - meanE) ** 2).sum()
    cellmass = np.bincount(bins[sup], weights=w[sup], minlength=G * G)
    cellsum = np.bincount(bins[sup], weights=(w * Efin)[sup], minlength=G * G)
    mcell = np.zeros(G * G)
    mcell[occ] = cellsum[occ] / cellmass[occ]
    ssw = (w * (Efin - mcell[bins]) ** 2).sum()
    r2cell = 1 - ssw / sst if sst > 0 else float("nan")
    return {"occ": occ_frac, "eff": eff, "varR": varR, "rough": rough, "r2cell": r2cell}


def linear_r2(y, P, L, A):
    y = np.asarray(y, np.float64).ravel()
    w = P.ravel()
    C0 = bs._indicator_covariance(P)
    mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j)) for j in range(L)])
    my = float((w * y).sum())
    var = float((w * (y - my) ** 2).sum())
    Y = y.reshape(P.shape)
    cross = np.zeros(L * A)
    for j in range(L):
        Py = P * Y
        cross[j * A:(j + 1) * A] = Py.sum(axis=tuple(k for k in range(L) if k != j)) \
            - mean[j * A:(j + 1) * A] * my
    lam, U = np.linalg.eigh(C0)
    keep = lam > 1e-10 * max(lam[-1], 1e-30)
    z = U[:, keep].T @ cross
    return float(cross @ (U[:, keep] @ (z / lam[keep])) / var) if var > 0 else float("nan")


def ipr(phi):
    phi = phi - phi.mean()
    return float((phi ** 4).sum() / ((phi ** 2).sum()) ** 2)


print(f"{'model':10s} {'ell':>6s} {'den':6s} {'occ':>6s} {'eff':>6s} {'varR':>5s} "
      f"{'rough':>6s} {'r2cell':>6s} {'gap':>8s} {'R2yE':>6s} {'linR2x':>7s} {'linR2y':>7s} {'IPR1':>7s} {'IPR2':>7s}")
for key in ["austen", "twitter2023", "bluesky2024", "pooled"]:
    P, lp, E = load(key)
    lam_med = bs._flux_median_slope(P.astype(np.float32), L, A)
    sup = P.ravel() > 0
    for mult in (0.5, 1.0, 2.0):
        ell = mult * lam_med
        for denom in ("degree", "P"):
            U, lamv, gap = laplacian_eigenmap(P, E, ell, denom)
            mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                                   for j in range(L)])
            coords = bs.coords_linear(U, mean, L, A)
            m = metrics(P, lp, E, coords)
            x, y = coords[0], coords[1]
            r2xE = float(np.corrcoef(x[sup], E[sup])[0, 1] ** 2)
            r2yE = float(np.corrcoef(y[sup], E[sup])[0, 1] ** 2)
            lx = linear_r2(x, P, L, A)
            ly = linear_r2(y, P, L, A)
            i1 = ipr(x[sup])
            i2 = ipr(y[sup])
            print(f"{key:10s} {ell:6.2f} {denom:6s} {m['occ']:6.3f} {m['eff']:6.1f} "
                  f"{m['varR']:5.2f} {m['rough']:6.3f} {m['r2cell']:6.3f} {gap:8.4f} "
                  f"{r2yE:6.3f} {lx:7.3f} {ly:7.3f} {i1:7.4f} {i2:7.4f}")
