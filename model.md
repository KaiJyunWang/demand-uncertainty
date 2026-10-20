# A continuous-time general-equilibrium menu cost model with observed quality

**Specification and implementation status.** Firms observe long-run product quality immediately when drawn at innovation and entry. Current log taste follows an OU diffusion; the sufficient firm state has three dimensions, with no posterior mean or variance. The active Julia solver implements this specification on a $101^3$ grid. Section 12 reports its checked finite-step equilibrium. All economic parameters are unchanged from the preceding illustrative calibration. Current outputs are in `results_observed/`.

## 1. Environment and information

Time is continuous. A representative household consumes a unit continuum of differentiated varieties, supplies labor, and owns all firms. Each firm produces one variety, chooses a nominal price, pays menu costs when repricing, and chooses cash-flow-financed R&D with a fixed operating expense. Firms can exit for outside value zero; each exit is matched by a distinct entrant, keeping the mass of varieties equal to one. An index $i$ denotes a product slot, occupied by successive firms.

Let $a_{it}$ denote current log taste and $\theta_{it}\equiv\bar a_{it}$ long-run log quality. The firm observes both. Between successful innovations, quality is constant and

$$
\boxed{da_{it}=(\theta_{it}-a_{it})dt+\sigma_a dZ_{it},\qquad d\theta_{it}=0.}
$$

Here $Z_{it}$ is an idiosyncratic standard Brownian motion, independent across firms. Thus log taste is an OU process conditional on quality. Its mean-reversion speed is one per year. Brownian taste shocks and future innovation arrivals and draws remain uncertain; the realized long-run quality does not need to be inferred. There is no sales-based Bayesian filtering, information acquisition, or posterior-variance dynamics.

Firms continuously observe current taste, either directly or from their own sales and known prices and aggregate absorption:

$$
a_{it}=\log q_{it}+\epsilon\log(p_{it}/P_t)-\log Y_t.
$$

All controls are adapted to these observations, known quality, and public aggregate paths. Idiosyncratic shocks and independent quality draws wash out under an exact law of large numbers, so aggregate paths are deterministic. There is no capital or government spending.

## 2. Household preferences and heterogeneous CES demand

The household maximizes

$$
\int_0^\infty e^{-\rho t}
\left[u(C_t)-\frac{\chi N_t^{1+\varphi}}{1+\varphi}\right]dt,
\qquad \rho,\chi>0,\quad \varphi>0,
$$

The consumption utility function is explicitly CRRA. The balanced-growth baseline uses its log case $\gamma=1$; the reason is derived in Section 8:

$$
u(C)=\begin{cases}
\dfrac{C^{1-\gamma}-1}{1-\gamma},&\gamma>0,\ \gamma\ne1,\\
\log C,&\gamma=1,
\end{cases}
\qquad u'(C)=C^{-\gamma}.
$$

Its consumption aggregator is

$$
C_t=\left[\int_0^1 e^{a_{it}/\epsilon}
(q^C_{it})^{(\epsilon-1)/\epsilon}\,di\right]^{\epsilon/(\epsilon-1)},
\qquad \epsilon>1,\quad a_{it}\in\mathbb R.
$$

Here $q^C_{it}$ denotes household consumption of variety $i$. The variable $a_{it}$ is log taste for good $i$, and $e^{a_{it}}>0$ is its taste weight. Different goods can have different taste weights even though the substitution elasticity $\epsilon$ is common. The exponent $1/\epsilon$ on the weight is a convention chosen so that demand is linear in $e^{a_{it}}$.

Cost minimization gives the exact consumption price index and demand:

$$
\boxed{P_t=\left[\int_0^1 e^{a_{it}}p_{it}^{1-\epsilon}\,di\right]^{1/(1-\epsilon)}},
\qquad
\boxed{q^C_{it}=e^{a_{it}}\left(\frac{p_{it}}{P_t}\right)^{-\epsilon}C_t}.
$$

For household purchases $q^C_{it}$, this demand formula gives $\int p_{it}q^C_{it}\,di=P_tC_t$. When menu services and R&D also use the same composite good, total sales replace $C_t$ by total composite absorption $Y_t$; see Sections 4 and 6. Taste weights enter both demand and the price index. They cannot be added to demand while retaining an unweighted price index.

## 3. Household budget, labor supply, and asset pricing

Let $W_t$ be the nominal wage, $B_t$ nominal bond holdings, $i_t$ the nominal interest rate, and $D_t$ aggregate nominal dividends net of menu and R&D expenses. The budget constraint is

$$
\dot B_t=i_t B_t+W_tN_t+D_t-P_tC_t.
$$

Impose a no-Ponzi condition and the usual household transversality condition. Bonds are in zero net supply. An interior household optimum satisfies

$$
\boxed{w_t\equiv\frac{W_t}{P_t}=\chi N_t^\varphi C_t^\gamma},
$$

$$
i_t=\rho+\gamma\frac{\dot C_t}{C_t}+\pi_t,
\qquad \pi_t\equiv\frac{\dot P_t}{P_t}.
$$

Because aggregate paths are deterministic, the real discount factor for firm payouts at $s\geq t$ is

$$
D^R_{t,s}=e^{-\rho(s-t)}\frac{u'(C_s)}{u'(C_t)},
\qquad
r_t=\rho+\gamma\frac{\dot C_t}{C_t}.
$$

Thus firms use the household's marginal-utility discount factor. Constant discounting at $\rho$ is valid in a stationary equilibrium with constant consumption, but generally not along a transition.

## 4. Revenue-funded menu costs and cash-flow-constrained R&D

### Production and expenditures

Each firm has linear technology $q_{it}=A n_{it}$, $A>0$, and real marginal cost $m_t=w_t/A$. Define $x_{it}=\log(p_{it}/P_t)$. A competitive composite-good sector combines varieties using the same CES technology and weights as household consumption. Its output serves consumption, menu services, and R&D. Let $Y_t$ be total composite absorption. Total variety sales and real revenue are

$$
q_{it}=e^{a_{it}-\epsilon x_{it}}Y_t,
\qquad R_t(x,a)=e^{a+(1-\epsilon)x}Y_t.
$$

Let $Q_t=\exp[M_t/(\epsilon-1)]$ denote the common quality scale. Each price change costs $Q_tK$ units of the composite good, or $P_tQ_tK$ nominally, where $K>0$ is fixed in quality-normalized units. The firm pays it from revenue, reducing its dividend by $P_tQ_tK$ at the adjustment. It is a fixed expense rather than a direct labor requirement or a percentage of instantaneous revenue. R&D effort $k_{it}\geq0$ is measured in quality-normalized composite goods per unit time. Its variable composite-good expenditure is $Q_tk_{it}$. Let $F_R>0$ be a fixed quality-normalized **operating cost per unit time** whenever $k_{it}>0$, and define $\ell_{it}=\mathbf{1}_{\{k_{it}>0\}}$. Total research expenditure is therefore $Q_t(k_{it}+F_R\ell_{it})$, deducted from revenue. The fixed operating expense uses the same composite good as variable research. It does not contribute to innovation intensity. These scaling conventions are required for the stationary balanced-growth version: keeping a fixed cost in unscaled composite units would make its relative burden vanish over time. Neither expense is rebated to the household: the purchased goods are used up in menu services and research.

The financing restriction applies to R&D flows, while menu payments retain the existing net-dividend settlement convention. Firms cannot borrow or obtain new equity to fund R&D, and past retained earnings do not augment the current R&D budget. We do not impose a cash-balance constraint on production or lump-sum menu payments. Consequently no retained-earnings state is introduced; this is an R&D financing constraint rather than a comprehensive prohibition on all external funding.

Real operating profit before these expenses is

$$
\Pi_t(x,a)=\left(e^x-\frac{w_t}{A}\right)e^{a-\epsilon x}Y_t.
$$

The cost $F_R$ is fixed with respect to the amount of positive effort, but is a **flow**, not a lump-sum fee each time research starts. It is paid only while research is active and disappears immediately at $k=0$. There is no separate research startup fee, R&D-capital stock, or active/inactive research-status state. Replacement entry is specified below and has no resource fee in this baseline. A one-time startup fee would require a different switching problem and is not assumed here. The menu cost $K$ remains a distinct lump-sum price-adjustment expense.

### Cash-flow financing constraint

Operating profit $\Pi_t$ is measured **before all R&D and menu expenses**. With linear production and no capital, depreciation, taxes, or working-capital timing, this is the cash flow available for research. Define normalized financing capacity

$$
\Pi_t^+(x,a)=\max\{\Pi_t(x,a),0\},\qquad
c_t^{\mathrm{RD}}(x,a)=\frac{\Pi_t^+(x,a)}{Q_t}.
$$

The pointwise cash-flow constraint becomes

$$
\boxed{k_{it}+F_R\mathbf{1}_{\{k_{it}>0\}}
\leq c_t^{\mathrm{RD}}(x_{it},a_{it}).}
$$

Equivalently the feasible effort set at capacity $c\geq0$ is

$$
\boxed{\mathscr K_F(c)=
\begin{cases}
\{0\},&c\leq F_R,\\
\{0\}\cup(0,c-F_R],&c>F_R.
\end{cases}}
$$

Thus a firm needs **strictly more** cash flow than the fixed operating expense to conduct any research. When active, its variable-effort ceiling is $\bar k_t^+(x,a)=[c_t^{\mathrm{RD}}(x,a)-F_R]_+$. All positive operating cash flow may fund total research expenditure, but not more. In particular, positive operating profits alone no longer imply that research is feasible. No external funds, past retained earnings, or new equity augment the current R&D budget.

For nonnegative operating profits, feasible research leaves the dividend flow nonnegative:
$\Pi_t-Q_t(k+F_R\ell)\geq0$. At negative operating profits, research is zero and the operating loss retains the existing household/net-dividend settlement; the firm may instead exercise its voluntary exit option for zero. This is optimal stopping rather than an automatic cash-flow default rule.

The menu payment $Q_tK$ has units of goods, whereas $\Pi_t$, $Q_tk$, and $Q_tF_R$ are goods per unit time. The lump-sum menu expense is still deducted from dividends at repricing and is not subtracted from the instantaneous research-flow budget. Recompute financing capacity at the new price. An innovation without repricing leaves current price, taste, and financing capacity unchanged on impact, but can change research participation through its effect on known quality and innovation value. Switching research on or off entails no additional lump-sum payment: $F_R$ is subsequently paid as a flow. Negative menu-payment impulses remain permitted under the retained settlement convention.

Between price adjustments $\dot x_{it}=-\pi_t$. At adjustment the firm chooses a new $x=\xi$ and pays the actual composite-good expense $Q_tK$; the action itself changes neither taste nor quality.

### Innovation technology and observed quality draws

The research intensity is

$$
\boxed{\lambda(k)=\lambda_0 k^\omega,\qquad \lambda_0>0,\quad 0<\omega<1.}
$$

Mean market log quality is the actual cross-sectional mean,

$$
M_t=\int\theta\,d\mu_t.
$$

At a successful innovation time $T$,

$$
\boxed{\theta_{iT}=M_{T^-}+\delta_Q+\sigma_Q\varepsilon_{iT},\qquad
\varepsilon_{iT}\sim\mathcal N(0,1),\quad \delta_Q>0,\quad\sigma_Q>0.}
$$

The firm observes the realized draw immediately. The normal distribution is a law of future draws, not a prior about an existing unobserved state. Draws are independent across firms and events and independent of Brownian taste shocks. A success changes quality but leaves current taste and nominal price unchanged; the firm may subsequently pay to reprice or choose to exit. A draw can improve or worsen the firm's old quality. Its expectation is above mean market **log** quality by $\delta_Q$.

For any function $f$, define the expected post-success operator

$$
\boxed{\mathcal B_t f(x,a,\theta)=
\int_{\mathbb R}f(x,a,M_t+\delta_Q+\sigma_Q\varepsilon)
\phi(\varepsilon)d\varepsilon,}
$$

where $\phi$ is the standard normal density. Before success, innovation value is an expectation over draws. After success, the firm conditions its actions on the realized quality. Replacing the integral by value at the mean draw would generally give a different model.

### Exit and replacement entry

Exit is irreversible with outside value zero. An incumbent receives no value from its replacement; exit cannot give that incumbent a free quality redraw. A distinct entrant receives the vacated slot. Entry is one-for-one replacement, with zero resource cost and no unrestricted free-entry zero-profit condition. Entry slots can carry rents.

The entrant draws and immediately observes

$$
\theta_t^e=M_t+\delta_Q+\sigma_Q\varepsilon^e,\qquad
\varepsilon^e\sim\mathcal N(0,1).
$$

Retain the previous entry-taste convention: $a_t^e=M_t+\delta_Q$, independently of the quality draw. Consequently initial taste need not equal realized long-run quality; it subsequently follows the same OU dynamics. This convention introduces no new parameter.

The entrant chooses its initial price for free **after observing its draw**:

$$
x_t^e(\theta)\in\arg\max_x V_t(x,M_t+\delta_Q,\theta).
$$

Its entry-state law is the pushforward of the normal draw:

$$
\boxed{\int h(s)\eta_t(ds)=\int_{\mathbb R}
h\!\left(x_t^e(M_t+\delta_Q+\sigma_Q\varepsilon),
M_t+\delta_Q,M_t+\delta_Q+\sigma_Q\varepsilon\right)
\phi(\varepsilon)d\varepsilon.}
$$

Every allocated slot becomes an operating entrant; there is no optional draw rejection. Positive entrant value almost surely and absence of immediate replacement chains must be checked. Free initial pricing and the option to operate without R&D support positive value at finite taste, wages, and prices. Subsequent price changes pay the usual menu cost.

## 5. Firm value, pricing, R&D, and exit

The sufficient state is $s=(x,a,\theta)$. Value is

$$
V_t(s)=\sup_{\{\tau_n,\xi_n\},k,\tau_X}\mathbb E_t\!\left[
\int_t^{\tau_X}D^R_{t,u}\big[\Pi_u-Q_u(k_u+F_R\mathbf1_{\{k_u>0\}})\big]du
-\sum_{n:t\leq\tau_n<\tau_X}D^R_{t,\tau_n}Q_{\tau_n}K\right].
$$

The terminal exit payoff is zero. Admissible policies satisfy the cash-flow cap, have no finite accumulation of price interventions or innovations, and have well-defined finite expected discounted payouts. Firms condition policies on observed quality. Aggregate paths and $M_t$ are taken as given.

Between innovations and interventions, the generator is

$$
\boxed{\mathcal L_t f=-\pi_t f_x+(\theta-a)f_a+
\frac{\sigma_a^2}{2}f_{aa}.}
$$

There is no quality diffusion or filtering covariance. Define the menu operator and normalized expected innovation gain by

$$
\mathcal M_t V(x,a,\theta)=\sup_\xi V_t(\xi,a,\theta)-Q_tK,
\qquad
\mathcal G_t(s)=\frac{\mathcal B_t V_t(s)-V_t(s)}{Q_t}.
$$

The QVI is

$$
\boxed{\max\left\{
\partial_t V_t+\mathcal L_tV_t+\Pi_t-r_tV_t
+Q_t\mathcal H_F(\mathcal G_t,c_t^{\mathrm{RD}}),
\mathcal M_tV_t-V_t,\;-V_t
\right\}=0,}
$$

where

$$
\mathcal H_F(d,c)=\sup_{k\in\mathscr K_F(c)}
\{-k-F_R\mathbf1_{\{k>0\}}+\lambda_0k^\omega d\}.
$$

For $c>F_R$ and $d>0$, the active candidate is

$$
k^u(d)=(\lambda_0\omega d)^{1/(1-\omega)},\qquad
k^a(d,c)=\min\{c-F_R,k^u(d)\},
$$

with net benefit

$$
A_F(d,c)=-F_R-k^a+\lambda_0(k^a)^\omega d.
$$

Choose $k^*=k^a$ only if $c>F_R$, $d>0$, and $A_F(d,c)>0$; otherwise choose $k^*=0$. Choose inactivity at indifference. The Hamiltonian is $\max\{0,A_F\}$ on this feasible positive-gain branch and zero otherwise. The active-branch KKT conditions are

$$
\lambda_0\omega(k^*)^{\omega-1}d-1-\zeta=0,\qquad
\zeta\geq0,\qquad\zeta(c-F_R-k^*)=0.
$$

The Inada condition $\lambda'(0^+)=\infty$ ensures positive conditional effort for a positive gain and adequate financing; the fixed expense still permits optimal inactivity. For $\omega=1/2$, slack-cap participation requires $d>2\sqrt{F_R}/\lambda_0$; with a binding cap it requires $\lambda_0(c-F_R)^\omega d>c$.

The post-success value in $\mathcal B_tV_t$ already includes the options to reprice at cost or exit after seeing the draw. Innovation does not provide a free price reset. Exit occurs at $V_t=0$, choosing exit at indifference. Conditional on survival, a paid reset has target

$$
x_t^*(a,\theta)\in\arg\max_\xi V_t(\xi,a,\theta).
$$

A single inaction band in price, if it exists, has quality- and taste-dependent boundaries and value matching $V_t(x,a,\theta)=V_t(x_t^*(a,\theta),a,\theta)-Q_tK$. A band structure and boundary smooth fit require justification; they are not imposed by the specification.

## 6. Distribution and aggregate equilibrium

Let $\mu_t$ be the unit-mass distribution of actual states $(x,a,\theta)$. Let $J_t(ds)$ be the paid price-adjustment outflow and $E_t(ds)$ the exit outflow, with $\nu_t=\int dJ_t$ and $e_t=\int dE_t$. For suitable test functions $h$,

$$
\boxed{\frac{d}{dt}\int h\,d\mu_t=
\int\big[\mathcal L_t h+\lambda(k_t^*)(\mathcal B_t h-h)\big]d\mu_t
+\int\big[h(x_t^*(a,\theta),a,\theta)-h(s)\big]dJ_t
-\int h(s)dE_t+e_t\int h\,d\eta_t.}
$$

The pricing and exit measures are endogenous intervention flows, possibly boundary fluxes rather than finite state-dependent hazards. They are supported on their optimal intervention sets. Taking $h=1$ verifies conservation of firm mass. If interventions have atoms in time, use the corresponding integrated measure equation and one-for-one replacement at those dates.

The CES index, production dispersion, and expenditures satisfy

$$
1=\int e^{a+(1-\epsilon)x}d\mu_t,\qquad
\Delta_t=\int e^{a-\epsilon x}d\mu_t,
$$

$$
\mathcal K_t=\int k_t^*d\mu_t,\quad
\mathcal A_t=\int\mathbf1_{\{k_t^*>0\}}d\mu_t,\quad
\mathcal E_t^{\mathrm{RD}}=\mathcal K_t+F_R\mathcal A_t,
$$

$$
\boxed{Y_t=C_t+Q_t(K\nu_t+\mathcal E_t^{\mathrm{RD}}),\qquad
N_t=Y_t\Delta_t/A.}
$$

Dividends obey

$$
D_t/P_t=Y_t-w_tN_t-Q_t(K\nu_t+\mathcal E_t^{\mathrm{RD}})
=C_t-w_tN_t.
$$

No cash-balance state, entry resource fee, or extra resource wedge is introduced. Market log quality and its growth are

$$
\boxed{M_t=\int\theta\,d\mu_t,\qquad
\dot M_t=\int\lambda(k_t^*)(M_t+\delta_Q-\theta)d\mu_t
+\int(M_t+\delta_Q-\theta)dE_t.}
$$

The exit-selection term uses actual observed quality, not an inferred expectation. New draws have mean $M_t+\delta_Q$ even though their subsequent policies depend on the realization. One-for-one replacement keeps the distribution normalized without a survival-renormalization term.

## 7. Taylor rule with a balanced-growth intercept

Let $g_C^\star$ denote the constant quality-adjusted consumption growth rate in the balanced-growth equilibrium. The interest-rate rule is

$$
\boxed{i_t=\rho+\gamma g_C^\star+\pi^*
+\phi_\pi(\pi_t-\pi^*)+u_t^m,\qquad \phi_\pi>1.}
$$

The natural real-rate intercept is $r^\star=\rho+\gamma g_C^\star$. Combining policy with the household Euler equation gives

$$
\gamma\left(\frac{\dot C_t}{C_t}-g_C^\star\right)
=(\phi_\pi-1)(\pi_t-\pi^*)+u_t^m.
$$

On the balanced-growth path with $u^m=0$, inflation equals $\pi^*$, consumption grows at $g_C^\star$, and $i=\rho+\gamma g_C^\star+\pi^*$. Both the growth rate and the consistent intercept are equilibrium outputs. If the old intercept $\rho+\pi^*$ were retained, the equilibrium would instead have $\pi=\pi^*+\gamma g_C^\star/(\phi_\pi-1)$ and would miss the inflation target.

This is an inflation-feedback specialization of a [Taylor rule](https://web.stanford.edu/~johntayl/Papers/Discretion.PDF). Inflation here is growth of the quality-adjusted CES price index. The rule does not target nominal prices of fixed physical varieties, which grow faster on the balanced-growth path. Initial nominal units remain a normalization. The Taylor principle alone does not prove determinacy for all transitions.

## 8. Stationary balanced-growth equilibrium

Seek $M_t=M_0+Gt$ and define

$$
Q_t=e^{M_t/(\epsilon-1)},\qquad g=G/(\epsilon-1),\qquad
\widetilde x=x-\log Q_t,\quad z=a-M_t,\quad q=\theta-M_t.
$$

Here $q$ denotes relative long-run log quality; variety quantities carry time/firm subscripts elsewhere. The stationary firm state is $(\widetilde x,z,q)$, with no fourth information state. Set

$$
C_t=Q_tc,\quad Y_t=Q_ty,\quad w_t=Q_t\widetilde w,\quad
V_t=Q_t\mathcal V(\widetilde x,z,q).
$$

For constant positive hours and a constant labor-disutility coefficient, the separable CRRA preferences require $\gamma=1$ when $g>0$:
$\widetilde w=\chi N^\varphi c^\gamma Q_t^{\gamma-1}$.
The maintained BGP baseline therefore uses log utility and
$\widetilde w=\chi N^\varphi c$.
The effective discount rate for normalized firm value is
$\widetilde r=\rho+(\gamma-1)g=\rho$.

### Stationary dynamics and innovation

Between interventions and innovations,

$$
\boxed{d\widetilde x=-(\pi^*+g)dt,\qquad
 dz=(q-z-G)dt+\sigma_a dZ,\qquad dq=-Gdt.}
$$

At success, $q$ is reset to the observed random value $\delta_Q+\sigma_Q\varepsilon$; $z$ and $\widetilde x$ are unchanged. The relative-quality drift $-G$ reflects unchanged own quality while market quality improves. The generator and expected redraw operator are

$$
\widetilde{\mathcal L}f=-(\pi^*+g)f_{\widetilde x}
+(q-z-G)f_z-Gf_q+\frac{\sigma_a^2}{2}f_{zz},
$$

$$
\boxed{\widetilde{\mathcal B}f(\widetilde x,z,q)
=\int f(\widetilde x,z,\delta_Q+\sigma_Q\varepsilon)
\phi(\varepsilon)d\varepsilon.}
$$

Normalized profits, financing capacity, and intervention values are

$$
\widetilde\Pi=(e^{\widetilde x}-\widetilde w/A)e^{z-\epsilon\widetilde x}y,
\quad c^{\mathrm{RD}}=[\widetilde\Pi]_+,
$$

$$
\widetilde{\mathcal M}\mathcal V=\sup_\xi\mathcal V(\xi,z,q)-K,
\qquad \mathcal G=\widetilde{\mathcal B}\mathcal V-\mathcal V.
$$

The stationary QVI is

$$
\boxed{\max\{\widetilde\Pi+\widetilde{\mathcal L}\mathcal V
-\rho\mathcal V+\mathcal H_F(\mathcal G,c^{\mathrm{RD}}),
\widetilde{\mathcal M}\mathcal V-\mathcal V,\;-\mathcal V\}=0.}
$$

### Entry, stationarity, and market clearing

An entrant has $z^e=\delta_Q$, $q^e=\delta_Q+\sigma_Q\varepsilon$, and free initial-price choice

$$
\widetilde x^e(q)\in\arg\max_\xi\mathcal V(\xi,\delta_Q,q).
$$

Thus

$$
\int h\,d\eta=\int h(\widetilde x^e(\delta_Q+\sigma_Q\varepsilon),
\delta_Q,\delta_Q+\sigma_Q\varepsilon)\phi(\varepsilon)d\varepsilon.
$$

The entry law is a distribution of observed qualities and draw-dependent prices, not a point mass at mean quality. If $\widetilde\mu$ is invariant and $e=\int dE$, its weak stationary equation is

$$
0=\int[\widetilde{\mathcal L}h+\lambda(k^*)(\widetilde{\mathcal B}h-h)]d\widetilde\mu
+\int[h(\xi^*(z,q),z,q)-h(s)]dJ
-\int h(s)dE+e\int h\,d\eta.
$$

Equilibrium also requires

$$
\begin{aligned}
1&=\int d\widetilde\mu, &1&=\int e^{z+(1-\epsilon)\widetilde x}d\widetilde\mu,\\
\Delta&=\int e^{z-\epsilon\widetilde x}d\widetilde\mu,&\nu&=\int dJ,\\
\mathcal E^{\mathrm{RD}}&=\int(k^*+F_R\mathbf1_{\{k^*>0\}})d\widetilde\mu,
&y&=c+K\nu+\mathcal E^{\mathrm{RD}},\\
N&=y\Delta/A,&\widetilde w&=\chi N^\varphi c,\\
0&=\int q\,d\widetilde\mu,&G&=\int\lambda(k^*)(\delta_Q-q)d\widetilde\mu
+\int(\delta_Q-q)dE.
\end{aligned}
$$

Zero mean relative quality ensures that $M_t$ is the actual market mean. The growth-flow equation follows from stationarity applied to $h=q$, so it is a diagnostic rather than an additional independent root condition. Solve for $(c,y,\widetilde w,G)$ using the price index, resources, labor supply, and zero mean relative quality, together with the QVI and invariant distribution.

Because innovation leaves current taste unchanged and entry initializes $z^e=\delta_Q$, the taste moment satisfies

$$
\boxed{\int z\,d\widetilde\mu=-G+\int(\delta_Q-z)dE.}
$$

Verify exit support in $\{\mathcal V=0\}$, equal entry and exit masses, and positive entrant values. Positive R&D, positive turnover, positive growth, existence, and uniqueness must be checked under the revised information structure rather than inherited from the old solution.

### Reconstruction

For $P_0=1$ if desired,

$$
P_t=P_0e^{\pi^*t},\qquad p_{it}=P_tQ_te^{\widetilde x_{it}},
\qquad a_{it}=M_t+z_{it},\quad\theta_{it}=M_t+q_{it}.
$$

Quality-adjusted consumption grows at $g$, the CES price index at $\pi^*$, and the cross-sectional nominal-price trend at $\pi^*+g$. Individual nominal prices remain constant between paid resets. Taste levels are $e^{a_{it}}$. The real interest rate is $\rho+g$ and the nominal rate $\rho+g+\pi^*$ in the log-utility BGP.

## 9. Interpretation

Known long-run quality predicts future taste even when current sales are unchanged. A success changes expected future demand immediately, but current profits and the research financing cap do not jump unless the firm also reprices. Policies can jump because the expected gain from the next redraw changes.

Higher current profits relax the research budget, but higher known quality can reduce the gain from replacing that quality with a new draw. Thus the cross-sectional profit–R&D relationship need not be monotone. A firm unable to cover $F_R$ cannot research at that state; Brownian taste movements, pricing decisions, and exit/entry still affect whether this inactivity persists. This is not an information trap.

With CES demand and linear production, the frictionless markup remains
$e^{x_t^{\mathrm{flex}}}=\epsilon w_t/[(\epsilon-1)A]$.
Quality affects demand dynamics and the stakes of price adjustment rather than this static markup formula.

## 10. Provisional calibration and references

Use one year as the time unit. The numerical equilibrium in Section 12 uses the preceding parameter vector without changing any economic primitive. These are **illustrative parameters, not empirical estimates**:

| Parameter | Starting value | Status |
| --- | --- | --- |
| Discount rate $\rho$ | $0.04$ | Menu-cost benchmark; [Golosov and Lucas (2007)](https://doi.org/10.1086/512625). |
| CRRA coefficient $\gamma$ | $1$ | Log utility required by the maintained positive-growth, constant-hours formulation. |
| CES elasticity $\epsilon$ | $7$ | Benchmark from [Golosov and Lucas (2007)](https://doi.org/10.1086/512625). |
| Inverse Frisch elasticity $\varphi$ | $5$ | Benchmark from [Galí (2018), calibration slides](https://crei.cat/wp-content/uploads/2016/07/basic_nk_may2018.pdf). |
| Inflation response $\phi_\pi$ | $1.5$ | Textbook benchmark in the same Galí slides. |
| Inflation target $\pi^*$ | $\log(1.02)$ | Maintained annual model target. |
| Productivity $A$ | $1$ | Units normalization. |
| Labor-disutility scale $\chi$ | $685.5749353654443$ | Fitted in the former learning-model equilibrium; held fixed in the observed-quality solve. |
| Menu cost $K$ | $0.02$ | Quality-normalized goods per adjustment; illustrative. |
| Fixed research flow $F_R$ | $0.01$ | Quality-normalized goods per year while active; illustrative. |
| Taste volatility $\sigma_a$ | $0.20$ | Illustrative; OU taste-gap standard deviation is $\sigma_a/\sqrt2$ conditional on permanently fixed quality. |
| Mean-reversion speed | $1$ | Maintained OU specification. |
| Innovation scale $\lambda_0$ | $3$ | Maintained user choice. |
| Innovation elasticity $\omega$ | $0.5$ | Curvature benchmark related to [Acemoglu et al. (2018)](https://doi.org/10.1257/aer.20130470); not a direct estimate for this goods-input technology. |
| Mean draw premium $\delta_Q$ | $0.20$ | Illustrative log-quality premium. |
| Draw standard deviation $\sigma_Q$ | $0.50$ | Actual realized quality dispersion, not posterior uncertainty. |
| Monetary disturbance $u_t^m$ | $0$ | Undisturbed BGP. |

There are no initial posterior-mean or posterior-variance parameters. For a transition, specify a joint initial distribution of actual $(x,a,\theta)$, observed by each firm. A provisional choice is $a_0\sim\mathcal N(-0.03,0.12^2)$ and $\theta_0\sim\mathcal N(-0.03,0.50^2)$ independently, with initial log nominal price zero. This is an illustrative nonstationary initial condition, with $M_0=-0.03$; a stationary BGP instead solves for its own joint distribution. An initial nominal-unit normalization can be imposed by rescaling all nominal prices together.

Menu and research parameters require joint fitting to price changes, sales persistence, R&D spending and participation, innovation outcomes, and entry/exit. Observed quality removes filtering parameters but does not identify innovation productivity or costs automatically.

## 11. Julia implementation and validation

The observed-quality solver is [src/ObservedQuality.jl](src/ObservedQuality.jl).
The default `scripts/solve_bgp.jl`, `scripts/plot_bgp.jl`, and
`scripts/recompute_bgp.jl` dispatch on `information="observed_quality"` in
[calibration_bgp.toml](calibration_bgp.toml).

The active grids have 101 points in each of the three state dimensions,
$101^3=1,030,301$ states. Price gaps span $[-0.35,0.30]$, with a dense core
$[-0.18,0.12]$ and four nodes in each tail. Taste gaps span $[-9,3.9]$ with
core $[-1.5,1.5]$; quality gaps span $[-8,3.6]$ with core $[-1.2,1.2]$.
Taste and quality each have 20 lower-tail nodes, 69 core nodes, and 12 upper-tail
nodes. These are grids of actual states, with no variance coordinate.

The discrete time step is $h=0.125$ years. Conditional on no success, exact OU
moments supply a five-node normal quadrature, deposited with nonnegative linear
interpolation. A success at step end has probability
$P_h(k)=1-\exp[-\lambda(k)h]$ and replaces actual relative quality with a
21-node normal quadrature. Incumbent taste and price stay continuous at this
redraw. Entry uses the same draw quadrature, choosing a free initial price
separately for each realization. The draw law is exogenous, even though the
realized state's policies depend on quality.

Howard policy iteration solves the pricing, research, and exit Bellman equation.
Matrix-free BiCGSTAB evaluates fixed policies and the invariant distribution.
The independent quality-redraw operator is factored as a rank-one operation in
quality, followed backward by the OU transition; the forward operator uses its
adjoint. Tests compare this factorization with the explicit joint transition.
Price-grid maximization uses a numerical softmax with temperature $10^{-5}$;
exit and the fixed-cost research participation decision remain explicit choices.
There is no research-choice entropy term.

The finite-step research objective is

$$
-h(k+F_R\mathbf1_{\{k>0\}})
+P_h(k)\,e^{-\rho h}(E_1\mathcal V-E_0\mathcal V),
$$

where $E_0$ is the no-success transition and $E_1$ includes a realized-quality
redraw. Active effort optimizes this probability subject to the exact cash-flow
cap, then compares its value with inactivity. The continuous-time Hamiltonian
in Section 5 is its small-$h$ limit.

For the aggregate root, the price grid is centered at
$\zeta=\log[\epsilon\widetilde w/((\epsilon-1)A)]$.
Conditional policies depend on $y,\widetilde w$ through the profit scale
$F=y\exp[(1-\epsilon)\zeta]$. Solving at the canonical frame
$\widetilde w_0=A(\epsilon-1)/\epsilon$, $y_0=F$ supplies a canonical CES index
$S=\int e^{z+(1-\epsilon)x_{gap}}d\widetilde\mu$ and dispersion
$D_0=\int e^{z-\epsilon x_{gap}}d\widetilde\mu$ after the price decision.
Recover

$$
\zeta=\frac{\log S}{\epsilon-1},\quad
y=FS,\quad c=y-K\nu-\mathcal E^{\mathrm{RD}},\quad
\widetilde w=\widetilde w_0e^\zeta,\quad
N=Fe^{-\zeta}D_0/A.
$$

A two-dimensional root in $(\log F,G)$ then imposes labor supply and zero mean
relative quality. This eliminates two aggregate conditions analytically; it
does **not** recalibrate $\chi$. All economic primitives are held fixed at the
values in Section 10. An independent check re-evaluates the full four aggregate
conditions, Bellman equation, invariant distribution, financing, research KKT
conditions, and growth moments in the recovered frame.

The invariant distribution is sampled before each price/exit decision; flow
aggregation uses the distribution after pricing and replacement entry. Report
continuous-time flow discrepancies separately from exact finite-step moment
errors. Neither a small root residual nor a larger state grid establishes
convergence to the continuous-time, continuous-state equilibrium.

Firm simulations use SciML's adaptive `RKMilCommute` SDE solver, with one
Brownian shock in current taste and deterministic quality between draws.
Interpolations.jl supplies multilinear policy/value-gain interpolation; research
is reoptimized at the exact current cash-flow cap at every solver stage.
Continuous callbacks locate repricing, integrated-intensity innovation events,
and exit with one-for-one entry. Quality draws are recorded and immediately
observed. The solver chooses its own time steps. The noise-free, single-innovation
level example instead uses adaptive `Tsit5` and suppresses subsequent innovations;
it remains a conditional path under the noisy equilibrium policies.

Core operator tests are in `test/observed_quality.jl` (also included in
`test/runtests.jl`). Separate `test/observed_simulation.jl` checks forced exit and
replacement, financing between grid nodes, observed realized entry quality,
and the analytical deterministic OU path. Current outputs and caches live in
`results_observed/`.

For reference, without innovation over a step $h$, the exact stationary-coordinate transition is

$$
q_{t+h}=q_t-Gh,\qquad
z_{t+h}=q_t+(z_t-q_t)e^{-h}-Gh
+\sigma_a\sqrt{(1-e^{-2h})/2}\,\varepsilon,
\quad\varepsilon\sim\mathcal N(0,1).
$$

The corresponding price drift is $\widetilde x_{t+h}=\widetilde x_t-(\pi^*+g)h$ absent repricing. These are transitions conditional on no innovation or intervention, not a finite-step substitute for the full controlled jump process.


## 12. Computed observed-quality equilibrium

Using Julia 1.13.0 from juliaup's default channel, the solver converges on the
$101\times101\times101$ grid with $h=0.125$ years and all economic parameters
held fixed, including $\chi=685.5749353654443$.

| Equilibrium quantity | Value |
| --- | --- |
| Detrended consumption $c$ | $0.3183718176$ |
| Detrended absorption $y$ | $0.3328116872$ |
| Detrended real wage $\widetilde w$ | $0.8644615391$ |
| Hours $N$ | $0.3307892982$ |
| Mean log-quality growth $G$ | $0.0790042768$ per year |
| Quality-adjusted consumption growth | $1.32545\%$ per year |
| CES inflation | $2\%$ per year |
| Nominal price trend | $3.35195\%$ per year |
| R&D-active firm mass | $42.7623\%$ |
| Firms unable to finance positive research, $c^{\mathrm{RD}}\leq F_R$ | $3.59454\%$ |
| Binding active research financing cap, share of all firms | $16.9066\%$ |
| Variable research expenditure | $0.0059788163$ |
| Fixed research expenditure | $0.0042762328$ |
| Total research expenditure | $0.0102550491$ |
| R&D expenditure / total absorption | $3.08134\%$ |
| Innovations per firm-year | $0.1482785321$ |
| Exit rate = entry rate, per firm-year | $0.0022750566$ |
| Price adjustments per firm-year | $0.2092410238$ |
| Expected entrant value | $0.8504993639$ |

Thus annual exit and replacement-entry flows are each $0.227506\%$ of the unit
firm mass. Positive research, turnover, and growth emerge from optimal policies;
none is imposed as an exogenous event rate. This is one positive-growth
numerical equilibrium, not a uniqueness result.

An independent check at the saved aggregates gives a maximum market-clearing
residual of $2.59\times10^{-6}$, Bellman residual $9.10\times10^{-12}$, and
invariant-distribution L1 residual $2.53\times10^{-11}$. The maximum research
financing violation is $3.47\times10^{-18}$, numerical roundoff, and the maximum
active-branch KKT residual is approximately $10^{-12}$.

Domain/quadrature clipping is $1.03\times10^{-10}$ of operating mass per step.
The fraction of exit flow at the lower quality boundary is
$3.23\times10^{-8}$; exit is overwhelmingly interior to that bound. The exact
split quality and taste moment errors are $2.50\times10^{-11}$ and
$1.75\times10^{-11}$. The corresponding continuous-time quality and taste
moment discrepancies are $-0.0001466$ and $-0.0045733$. These nonzero discrepancies
reflect the finite-step approximation, despite the small equilibrium residuals.
Time-step and further state/quadrature refinement are needed for quantitative
continuous-time convergence claims.

Machine-readable results are [solution.csv](results_observed/solution.csv) and
[balanced_growth.toml](results_observed/balanced_growth.toml); independent checks
are [checks.toml](results_observed/checks.toml). The figures show the three
[marginal distributions](results_observed/figures/equilibrium_marginals.png),
[pricing, exit, and research policy slice](results_observed/figures/exit_policy.png),
and simulated [firm paths](results_observed/figures/firm_paths.png) and
[absolute levels](results_observed/figures/firm_levels.png). Separate 30-year
figures are in `results_observed/figures/30_year/`. Successful-innovation level
paths with subsequent noise shut down are in
`results_observed/figures/innovation_levels/`.

The adaptive 300-year simulation (seed 2026) records 41 innovations and one exit
followed by entry. The 30-year simulation records five innovations and no exits.
Both remain inside the policy grid, with cash-flow violations only at roundoff
scale. The separate noise-free innovation path uses 47 accepted ODE steps and
keeps realized long-run quality constant after its initial observed draw.
The core regression suite and all 48 new operator/simulation checks pass; logs
are saved as `results_observed/core_tests.log` and
`results_observed/simulation_tests.log`.
