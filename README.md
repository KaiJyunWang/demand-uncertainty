# Menu costs, learning, innovation, and balanced growth

The model is specified in [model.md](model.md). The active solver finds a stationary
**balanced-growth equilibrium**: mean log quality increases at a constant rate,
quality-adjusted consumption grows, and the firm distribution is stationary in
coordinates relative to average quality.

## Run

Use the juliaup default `julia` command. The current default and project minimum are Julia 1.13.0 / 1.13; only standard libraries are used:

```sh
julia --project=. test/runtests.jl
JULIA_NUM_THREADS=16 julia --project=. scripts/solve.jl calibration_bgp.toml
```

`scripts/solve_bgp.jl` is the same entry point. In a restricted environment, prefix
commands with `JULIA_DEPOT_PATH=/tmp/demand-julia-depot` for a writable compilation
cache. No dependency download is needed.

The calibration uses log CRRA consumption utility (`gamma=1`), which permits
constant hours with the specified separable labor disutility. Menu cost `K` and
R&D effort `k` are in **quality-normalized composite-good units**; actual real
expenses are `Q(t)*K` and `Q(t)*k`. These conventions are necessary for the chosen
stationary balanced-growth formulation, and are explicit in model.md. They change
the earlier fixed-unscaled-cost, `gamma=2` transition specification.

## Equilibrium and algorithm

Write `M(t)=M(0)+G*t` and `Q(t)=exp(M(t)/(epsilon-1))`. The unknowns are detrended
consumption `c`, absorption `y`, wage `wtilde`, and mean-log-quality growth `G`.
The stationary state is `(x-log(Q), a-M, b-M, v)`. The numerical price grid is
further centered on the flexible-price markup, so `xmin` and `xmax` bound a price
gap relative to that center. The other bounds apply to taste and belief gaps.

For each candidate aggregate tuple:

1. Sparse Howard policy iteration solves pricing and R&D jointly.
2. The induced row-stochastic Markov matrix supplies an invariant distribution.
3. Four residuals impose the CES index, goods clearing, labor supply, and zero mean
   belief relative to market quality. Stationarity then supplies the growth-flow
   identity, which is reported separately as a numerical diagnostic.

If `g=G/(epsilon-1)`, quality-adjusted consumption and wages grow at `g`, the CES
price index grows at `pi_target`, and the cross-sectional nominal-price trend is
`pi_target+g`. The Taylor intercept includes `rho+gamma*g`, ensuring that growing
consumption does not push equilibrium inflation away from its target. There is
no terminal consumption restriction or frozen-quality tail in the BGP solver.

The observation transition uses exact conditional Gaussian filter moments and
positive 3-by-3 quadrature with multilinear interpolation. R&D success is split to
the end of each time step, with probability `1-exp(-lambda(k)*dt)`, and resets
beliefs to `(premium,sigma_Q^2)`. R&D optimizes this discrete-step probability,
converging to the continuous-time policy as `dt` decreases.

The default `price_temperature` smooths the grid-price maximum. Its Bellman error
is at most `price_temperature*log(nx)` per step. Pricing probabilities in the
forward equation match the backward objective, including its numerical entropy
term. Setting temperature to zero restores deterministic price-grid choices.

## Full 51-point grids

The active BGP configuration has exactly 51 nodes in each of the four state
dimensions, or 6,765,201 total states. Large grids automatically use
`src/LargeBGP.jl` rather than assembling/factorizing a global transition matrix.
The observation transition is factored into a price interpolation and a sparse
three-dimensional quality/belief/variance kernel. Pricing softmax blocks are
represented as a diagonal term plus a rank-one term. These operations reproduce
the small-grid discrete model, including its covariance, R&D choices, and paid
price changes; they do not reduce the number of states.

Matrix-free BiCGSTAB evaluates Howard policies and computes the invariant
probabilities. The default 16-thread command above is appropriate for the current
host. Set the thread count to suit another machine. Large runs log firm and linear
iterations and save accepted aggregate progress in `large_progress.toml`.

## Results and diagnostics

The solver writes to `results_bgp/`:

- `solution.csv` (also copied to project-root `bgp_solution_51.csv` for the 51-point CLI run): a single row of aggregate equilibrium values, growth and interest
  rates, convergence diagnostics, actual grid counts and Julia version.
- `balanced_growth.toml`: equilibrium and growth rates, resource residuals,
  Bellman and invariant-distribution residuals, boundary clipping and growth-flow
  diagnostics.
- `distribution.csv`: state coordinates and policies for the small-grid backend.
  The large-grid backend reports aggregate results in `solution.csv` without
  automatically writing millions of state-level CSV rows.
- `checkpoint.toml`: the four aggregate unknowns, residuals and root history.

A failed root search exits with code 2 and saves candidate outputs. A converged
root still needs checks of state-grid, domain, time-step, and price-temperature
accuracy. Boundary points are clamped rather than leaking probability. Growth-flow
errors reveal the effects of this truncation and the discrete event splitting.

A checkpoint can restart any BGP grid:

```sh
julia --project=. scripts/solve_bgp.jl calibration_bgp.toml results_bgp/checkpoint.toml
julia --project=. scripts/refine.jl calibration_bgp.toml
```

The second command runs independent time-step, grid, domain, and price-temperature
sensitivity cases. These can be substantially more expensive than a single run.
Provisional numerical results and remaining accuracy limits are recorded in
model.md. The parameter levels are illustrative, not an estimated calibration.

## Files and legacy routines

`src/BalancedGrowth.jl` implements the small-grid stationary solver;
`src/LargeBGP.jl` implements its factorized full-grid counterpart;
`src/DemandUncertainty.jl` contains its shared primitives and transition kernels.
`test/runtests.jl` covers filtering, R&D, positive transitions, mass conservation,
backward/forward accounting, growth transformations and a complete BGP solve.

The earlier finite-horizon routines remain available through
`scripts/solve_transition.jl calibration.toml` and `scripts/refine_transition.jl`.
They concern the previous level-cost specification and are not the balanced-growth
solver or a validated solution of the growth model. Use the BGP entry point above
for stationary long-run analysis.

The completed 51-point report is [bgp_solution_51.csv](bgp_solution_51.csv). On
Julia 1.13.0 its maximum equilibrium residual is `6.56e-7`, with annual
quality-adjusted consumption growth `0.05148%` and CES inflation `2%`. Raw growth
rates are continuous per-year rates; the `_pct_per_year` columns apply `100*expm1`.
`C`, `Y`, and `w` are detrended levels. The input snapshot and full iteration log
are saved in `results_bgp/` alongside the report.

Plot the stationary solution and simulate one firm using CairoMakie:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=. scripts/plot_bgp.jl calibration_bgp.toml 300 2026
```

The last two arguments are the simulation horizon in years and the random seed.
The script reads `checkpoint.toml` from the configured output directory. Its first
run reconstructs the full-grid firm problem and invariant distribution at those
aggregates, checks equilibrium residuals, and caches values and masses in
`plot_state.jls`. Subsequent runs reuse the cache; remove it after changing solver
code or calibration. This does not use the older coarse `distribution.csv`.

Outputs in `results_bgp/figures/` are `equilibrium_marginals.{png,pdf}`,
`firm_paths.{png,pdf}`, `marginals.csv`, `firm_path.csv`, `firm_events.csv`, and
`diagnostics.toml`. The four marginals remain the solver's **pre-decision finite-grid
probability masses**. Prices are shifted back to detrended log relative prices.

Firm simulation uses a SciML `SDEProblem` and adaptive `RKMilCommute`, with absolute
tolerance `1e-7` and relative tolerance `1e-5`. No integration step, maximum step,
or fixed observation grid is imposed: saved observations include the solver's
accepted steps and both sides of events. The two filtered shocks share one
Brownian motion. `Interpolations.jl` provides multilinear R&D and intervention
surplus policies on the four-dimensional grid, and a three-dimensional reset
price policy. Flat extrapolation is used for policy evaluations outside the grid;
continuous states are not clipped, and outside-domain observations are counted
in diagnostics.

Price adjustment is a continuous callback at an upward crossing of the
unsmoothed intervention surplus (best continuation value minus menu cost minus
current-price continuation value). Reset prices maximize the numerical
continuation value at grid nodes and are linearly interpolated between quality
states. This extracts an impulse policy from the existing finite-time-step,
softmax-smoothed equilibrium solution; it does not reinterpret quarterly choice
probabilities as continuous-time arrival rates. The equilibrium policies and
marginals retain their original discretization error; the continuous-time sample
need not have exactly the same invariant law as that finite-grid approximation.

Innovation arrives when integrated `lambda0*k^omega` reaches an independently
drawn exponential threshold. At that instant, the belief and variance reset to
`premium` and `sigma_Q^2`, respectively; taste is continuous, and the price policy
is checked again. Dashed plot markers show these actual event times. The band
shows posterior uncertainty about latent detrended quality. The path CSV also
contains log nominal price, which is constant between price adjustments,
cumulative menu payments, and cumulative net cash. Menu costs are instantaneous
cash payments, so the profit-flow panel subtracts only R&D; cumulative net cash
subtracts both R&D and menu costs. Initial states are drawn from the solved
finite-grid stationary distribution. The 300-year default helps display rare
innovations without changing their intensity.

Run simulation checks with:

```bash
julia --threads=4 --project=. test/plot_diagnostics.jl
```

The active 51-node price grid spans `[-0.075, 0.075]` around the flexible-price
center, with spacing `0.003`. The previous range was `[-0.45, 0.45]` with spacing
`0.018`. Earlier result directories have been removed; current comparison reports
retain the before/after summary. Economic
parameters, the three other 51-node grids, and the equilibrium time step are
unchanged. To recompute the current equilibrium:

```bash
julia --threads=16 --project=. scripts/solve_bgp.jl calibration_bgp.toml
julia --threads=16 --project=. scripts/plot_bgp.jl
```

The optional third solver argument transfers values and probability mass to the
new price grid as an initial guess. The solver subsequently recomputes policies,
the invariant distribution, and aggregate market clearing. It saves the converged
state in `plot_state.jls` so plotting does not repeat the firm solve. The aggregate
report includes price-grid spacing, occupied/effective node counts, boundary
mass, and price-drift clipping. `price_grid_comparison.csv` and
`figures/price_grid_comparison.{png,pdf}` compare the old and new equilibria.

`firm_levels.{png,pdf}` additionally plots the simulated firm's nominal price,
its relative price, undetrended taste state, and demand multiplier. Normalize
`P(0)=1` and `M(0)=0`, so `P(t)=exp(pi*t)`, `M(t)=G*t`, and `Q(0)=1`.
The reconstruction is `p_i(t)=exp(x_tilde(t)+(pi+g)*t)`,
`a_i(t)=taste_gap(t)+G*t`, and the demand multiplier is `exp(a_i(t))`.
The nominal-price panel uses a logarithmic axis. The firm's initial price is its
equilibrium draw, rather than being separately forced to one. These level series
are included in `firm_path.csv`; the original detrended figures are retained.

Large simulation CSVs and binary solver caches are excluded from Git. Run
`scripts/plot_bgp.jl` to regenerate them. The optional grid-comparison script
requires an independently saved baseline directory; earlier baselines are not
included in this repository.
