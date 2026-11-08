"""Checks for 'Momentum and batch size in streaming least squares'.

1. Symbolic: the noiseless per-mode determinant factors as (mu - beta)((mu + beta)^2 - mu k^2); the noise enters through the
   (1,1) cofactor mu (mu + beta); the generating function of the squared impulse response is mu (mu + beta) / p0(mu).
2. Symbolic: the two endpoint polynomials of Lemma 5 and their positivity on (0, 1].
3. Numerical: the second-moment map is built directly from the recursion; on random spectra
   (a) every stable heavy-ball setting obeys Lemma 3, (b) SGD at eta0 obeys Lemma 4,
   (c) the explicit heavy-ball tuning has spectral radius at most 1 - delta/4, and the resulting speedup is at least
       (1/16) min(max(1, B h_max / tr H), sqrt kappa) against the SGD ceiling.
usage: python checks.py [trials]"""
import sys
import numpy as np
import sympy as sp

# ---------- 1. symbolic identities ----------
mu, be, k, eps = sp.symbols('mu beta k epsilon')
M = sp.Matrix([[k**2 + eps, -2*be*k, be**2], [k, -be, 0], [1, 0, 0]])
p = sp.expand((mu*sp.eye(3) - M).det())
p0 = (mu - be)*((mu + be)**2 - mu*k**2)
assert sp.expand(p - (p0 - eps*mu*(mu + be))) == 0
# squared impulse response of r^2 - k r + beta: g(0)=1, g(1)=k, g(n)=k g(n-1) - beta g(n-2);
# check sum_n g(n)^2 z^n = (1 + beta z) / ((1 - beta z)((1 + beta z)^2 - k^2 z)) by series to order 12
z = sp.symbols('z')
g = [sp.Integer(1), k]
for n in range(2, 14):
    g.append(sp.expand(k*g[-1] - be*g[-2]))
ser = sum(g[n]**2 * z**n for n in range(13))
gf = (1 + be*z)/((1 - be*z)*((1 + be*z)**2 - k**2*z))
diff = sp.series(gf, z, 0, 13).removeO() - ser
assert sp.expand(diff) == 0
print("1. determinant factorisation, cofactor and squared-impulse generating function: OK")

# ---------- 2. endpoint polynomials ----------
d, x = sp.symbols('delta x', positive=True)
m0 = 1 - d/4; b0 = (1 - d/2)**2; k0 = 1 + b0 - x
G = sp.expand((m0 - b0)*((m0 + b0)**2 - m0*k0**2) - m0*(m0 + b0)*x*d/2)   # > 0  <=>  Q(mu0) > delta x / 2
assert sp.simplify(sp.diff(G, x, 2) + 2*m0*(m0 - b0)) == 0
for name, xv in (("x = delta^2/4", d**2/4), ("x = 1", sp.Integer(1))):
    poly = sp.Poly(sp.expand(G.subs(x, xv)), d)
    roots = [float(r) for r in sp.real_roots(poly) if 0 < r <= 1]
    assert not roots and poly.eval(sp.Rational(1, 2)) > 0
    print(f"2. G at {name}: {sp.factor(poly.as_expr())}; no root in (0, 1], positive at 1/2")


# ---------- 3. numerics on the second-moment map ----------
def moment_map(h, eta, beta, B):
    """Linear map on (a_j, x_j, c_j) = (E e_j(t)^2, E e_j(t) e_j(t-1), E e_j(t-1)^2), Gaussian rows, batch B."""
    dd = len(h); kk = 1 + beta - eta*h
    T = np.zeros((3*dd, 3*dd))
    for j in range(dd):
        a, xx, c = 3*j, 3*j + 1, 3*j + 2
        T[a, a] += kk[j]**2 + eta**2*h[j]**2/B
        T[a, xx] += -2*beta*kk[j]
        T[a, c] += beta**2
        for l in range(dd):
            T[a, 3*l] += eta**2*h[j]*h[l]/B
        T[xx, a] += kk[j]; T[xx, xx] += -beta
        T[c, a] += 1.0
    return T


def radius(h, eta, beta, B):
    return max(abs(np.linalg.eigvals(moment_map(h, eta, beta, B))))


rng = np.random.default_rng(20261108)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
wA, nA, wB, wC, wS = 0.0, 0, np.inf, np.inf, np.inf
for _ in range(N):
    dd = int(rng.integers(2, 25)); h = np.sort(10**rng.uniform(-3.5, 0, dd))[::-1]; B = int(10**rng.uniform(0, 4))
    tr, hmax, hmin = h.sum(), h[0], h[-1]; kap = hmax/hmin
    beta = 1 - 10**rng.uniform(-3, 0); eta = rng.uniform(0, 2*(1 + beta)/hmax)
    r = radius(h, eta, beta, B)
    if r < 1:
        nA += 1; wA = max(wA, (1 - r)/min(8*B*hmin/tr, 4/np.sqrt(kap)))
    e0 = min(B/(4*tr), 1/(2*hmax))
    wB = min(wB, (1 - radius(h, e0, 0.0, B))/(0.75*e0*hmin))
    delta = min(1.0, 2/np.sqrt(kap), B*hmin/tr)
    rh = radius(h, delta**2/(4*hmin), (1 - delta/2)**2, B)
    wC = min(wC, (1 - rh)/(delta/4))
    ceil = 4*hmin*min(B/tr, 1/hmax)
    wS = min(wS, ((1 - rh)/ceil)/min(max(1, B*hmax/tr), np.sqrt(kap)))
print(f"3a. Lemma 3 over {nA} stable heavy-ball points: max (1 - rho)/bound = {wA:.4f}  (<= 1)")
print(f"3b. Lemma 4 over {N} SGD points: min (1 - rho)/((3/4) eta0 h_min) = {wB:.4f}  (>= 1)")
print(f"3c. Lemma 5 over {N} points: min (1 - rho)/(delta/4) = {wC:.4f}  (>= 1)")
print(f"3d. speedup / min(max(1, B h_max/tr H), sqrt kappa) >= {wS:.4f}  (theorem: 1/16 = 0.0625)")
