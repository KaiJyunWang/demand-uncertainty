# Menu costs, observed quality, innovation, and balanced growth

The current specification is [model.md](model.md). Firms observe long-run quality
immediately when drawn; current log taste follows an OU diffusion. The firm state
has three dimensions: price, current taste, and known long-run quality.
There is no Bayesian filter or posterior-variance state.

The general equilibrium includes CES demand, log utility, labor supply, a Taylor
inflation rule, revenue-funded menu costs, cash-flow-constrained R&D with a fixed
operating expense, endogenous exit with zero outside value, and one-for-one
replacement entry. Both innovators and entrants draw from the innovation
quality distribution. Entrants select their first price after observing quality.

## Run

Use juliaup's default `julia`; the requested runtime is Julia 1.13.0.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=. test/runtests.jl
julia --threads=4 --project=. test/observed_simulation.jl
julia --threads=16 --project=. scripts/recompute_bgp.jl calibration_bgp.toml
```

The last command solves, independently checks the saved equilibrium, and exports
figures and adaptive 300-year and 30-year firm simulations. Separate commands are:

```sh
julia --threads=16 --project=. scripts/solve_bgp.jl calibration_bgp.toml
julia --threads=16 --project=. scripts/check_observed_bgp.jl calibration_bgp.toml
julia --threads=16 --project=. scripts/plot_bgp.jl calibration_bgp.toml
```

For a writable compilation cache in a restricted environment, prefix commands
with `JULIA_DEPOT_PATH=/tmp/demand-julia-depot:/home/b11303072/.julia`.

## Numerical method

`src/ObservedQuality.jl` implements the three-state solver. The active grids have
101 nodes in each dimension, totaling 1,030,301 states. Dense central grids and
explicit tails resolve price interventions, weak firms, and exit. Between events,
the taste transition uses exact OU moments and positive normal quadrature.
Actual quality is constant in levels between innovations. Innovation integrates
over normally distributed realized qualities instead of resetting all firms to
the mean draw. Free initial prices are optimized separately for each entry draw.

The solver uses matrix-free Howard iteration and an invariant-distribution solve.
It factors the independent quality redraw to avoid a dense joint transition.
Backward and forward operators are adjoints. Research has an explicit global
participation comparison and is capped by current operating profit after its
fixed expense. Incumbent exit value excludes replacement-firm value.

A two-dimensional root in operating-profit scale and quality growth solves the
same equilibrium as the four aggregate conditions: the CES price normalization
and resource condition are eliminated analytically. All economic primitives,
including household labor-disutility scale `chi`, are held fixed. Labor supply
and zero mean relative actual quality remain root conditions.

The finite-step model places innovations at step end, with probability
`1-exp(-lambda(k)*dt)`. Research optimizes this probability. Default `dt=0.125`
years and price smoothing `1e-5` are numerical approximations; time-step, domain,
quadrature, and grid sensitivity matter for quantitative claims.

## Outputs

Current results are written to `results_observed/`, including `solution.csv`,
`balanced_growth.toml`, `checkpoint.toml`, `checks.toml`, `run_calibration.toml`,
and a regenerable `observed_state.jls` cache. Figures and small marginal/event
CSVs are under `results_observed/figures/`; large firm-path CSVs and binary caches
are excluded from Git. Full state-level distribution CSVs are not automatically
written. The cache records a hash of the configuration and solver source.

The figures show all three equilibrium marginals and firm price, taste, known
quality, R&D, and profit paths. Absolute price levels use `P₀=1`. Simulations use
SciML's adaptive `RKMilCommute` SDE solver without a prescribed time step and
Interpolations.jl policies. Research is reoptimized at the exact cash-flow cap
at each solver stage. Realized quality is drawn and recorded on every innovation
and entry. The innovation-level example shuts down subsequent Brownian noise
and innovation arrivals; it uses adaptive `Tsit5` and is a conditional firm path in the noisy equilibrium.

## Regression fixtures

The tests retain small calibration fixtures in `test/fixtures/` for regression
coverage of shared numerical routines. Published documentation and outputs
cover the current observed-quality model only.

## Current equilibrium

With all economic parameters held fixed, the 101³ solve has 42.76% of firms
conducting R&D, R&D equal to 3.081% of absorption, annual exit and entry each
equal to 0.2275%, and annual quality-adjusted consumption growth of 1.3254%.
The maximum equilibrium residual is 2.59e-6. See
[solution.csv](results_observed/solution.csv),
[checks.toml](results_observed/checks.toml), and
[model.md, section 12](model.md#12-computed-observed-quality-equilibrium).
These are finite-step numerical results at illustrative parameters.
