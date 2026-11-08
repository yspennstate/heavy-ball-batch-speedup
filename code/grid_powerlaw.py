"""Exact mean-square spectral radius of heavy-ball SGD on realizable streaming
least squares, Gaussian x ~ N(0, H), H = diag(h), h_j = j^-alpha (j = 1..d), mini-batch B.
Error e_t = w_t - w*; e_{t+1} = K_t e_t - beta e_{t-1}, K_t = (1+beta) I - eta Hhat_t, Hhat_t the batch covariance.
For Gaussian x, E[Hhat A Hhat] = H A H + (1/B)(H A H + tr(H A) H), which keeps diagonal matrices diagonal, so with
a = diag E e_t e_t', x = diag E e_t e_{t-1}', c = diag E e_{t-1} e_{t-1}' and kb = 1 + beta - eta h the second moments obey exactly
    a' = kb^2 a + (eta^2/B)(h^2 a + (h.a) h) - 2 beta kb x + beta^2 c,   x' = kb a - beta x,   c' = a,
a linear map on R^{3d}. Its spectral radius rho(eta, beta, B) is the mean-square contraction per step; beta = 0 is plain SGD.
We minimize rho over (eta, beta) on a grid, and report the slowest-mode speedup (1 - rho_HB) / (1 - rho_SGD).
usage: python grid_powerlaw.py d alpha B_list   (e.g. 100 2.0 1,10,100,1000,100000)"""
import sys, numpy as np


def rho(h, eta, beta, B):
    d = len(h); kb = 1 + beta - eta * h
    A = np.zeros((3 * d, 3 * d)); I = np.arange(d)
    A[I, I] = kb ** 2 + eta ** 2 / B * h ** 2
    A[:d, :d] += eta ** 2 / B * np.outer(h, h)
    A[I, d + I] = -2 * beta * kb; A[I, 2 * d + I] = beta ** 2
    A[d + I, I] = kb; A[d + I, d + I] = -beta
    A[2 * d + I, I] = 1.0
    return max(abs(np.linalg.eigvals(A)))


def best(h, B, betas):
    out = (2.0, None, None)
    for beta in betas:
        lo, hi = -6.0, np.log10(2 * (1 + beta) / h[0])     # log10 eta search range
        etas = np.logspace(lo, hi, 60)
        r = [rho(h, e, beta, B) for e in etas]
        i = int(np.argmin(r))
        # refine around the grid minimum
        for e in np.logspace(np.log10(etas[max(i - 1, 0)]), np.log10(etas[min(i + 1, 59)]), 25):
            v = rho(h, e, beta, B)
            if v < out[0]: out = (v, e, beta)
    return out


if __name__ == '__main__':
    d, alpha = int(sys.argv[1]), float(sys.argv[2]); Bs = [int(v) for v in sys.argv[3].split(',')]
    h = np.arange(1, d + 1) ** -alpha
    print(f"d {d} alpha {alpha}: kappa = {h[0] / h[-1]:.3g}, tr H = {h.sum():.3f}")
    betas = np.concatenate([[0.0], 1 - np.logspace(-3, 0, 40)[:-1]])
    for B in Bs:
        sgd = best(h, B, [0.0]); hb = best(h, B, betas)
        print(f"  B={B:6d}  SGD rho {sgd[0]:.8f} (eta {sgd[1]:.3g})   HB rho {hb[0]:.8f} (eta {hb[1]:.3g}, beta {hb[2]:.4f})"
              f"   speedup (1-rho_HB)/(1-rho_SGD) = {(1 - hb[0]) / (1 - sgd[0]):.3f}   sqrt(kappa) bound {np.sqrt(h[0] / h[-1]):.1f}",
              flush=True)
