# Heavy-ball momentum and batch size in streaming least squares

Yitzchak Shmalo

Paper: [paper/momentum_batch.pdf](paper/momentum_batch.pdf) (source [paper/momentum_batch.tex](paper/momentum_batch.tex)).

For realizable streaming least squares with Gaussian covariates $x\sim N(0,H)$ and mini-batches of size $B$, the paper
compares heavy-ball momentum with plain SGD, each tuned to minimise the spectral radius $\rho$ of the exact
second-moment recursion. With $\kappa=h_{\max}/h_{\min}$ and $m=\min(\max(1,Bh_{\max}/\mathrm{tr}\,H),\sqrt\kappa)$ it proves

$$\max(1,m/16)\le\frac{1-\rho^*_{\mathrm{HB}}}{1-\rho^*_{\mathrm{SGD}}}\le\frac{128}{3}\,m .$$

The lower bound comes from an explicit choice of parameters: $\delta=\min(1,2/\sqrt\kappa,Bh_{\min}/\mathrm{tr}\,H)$,
$\sqrt\beta=1-\delta/2$, $\eta=\delta^2/(4h_{\min})$, which gives $\rho\le1-\delta/4$. The upper bound on the
heavy-ball gap is Theorem 1 of Morwani, Meterez, Nair and Kakade (arXiv:2606.19179). The paper also gives the exact
mean-square stability region of stochastic heavy ball in this model.

## Code

Python 3.14 with NumPy 2.4 and SymPy 1.14.

- `code/checks.py [trials]`: builds the $3d\times3d$ second-moment map from the recursion, checks symbolically the
  determinant factorisation, the generating function of the squared impulse response and the two endpoint
  polynomials of Lemma 5.2, and checks Lemmas 4.1, 4.2 and 5.2 and the speedup bound on random spectra.
  The paper used `2000` (`code/checks_run.txt`).
- `code/grid_powerlaw.py d alpha B_list`: minimises $\rho$ over an $(\eta,\beta)$ grid for $h_j=j^{-\alpha}$ and
  prints the optimal speedup. Table 1 of the paper comes from `100 2.0`, `50 2.0`, `100 1.5` and `100 1.0`
  with `B_list` = `1,10,100,1000,100000` (`code/grid_powerlaw_run.txt`).
