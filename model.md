# A continuous-time general-equilibrium menu cost model

## 1. Environment

Time is continuous. A representative household consumes a unit continuum of differentiated goods indexed by $i\in[0,1]$, supplies labor, and owns all firms. Each monopolistically competitive firm produces one good and pays a fixed menu expense out of revenue whenever it changes its nominal price, and chooses R&D expenditure to reset its latent quality. Goods have different, time-varying taste weights in household preferences.

The baseline has idiosyncratic taste shocks and deterministic aggregate paths. An exact law of large numbers eliminates aggregate sampling uncertainty. The price index, real wage, consumption, employment, and distribution of relative prices are equilibrium objects. A Taylor interest-rate rule targets long-run inflation. The balanced-growth specification in Section 9 is the main long-run equilibrium: average log quality grows, while detrended firm states and aggregates are stationary. There is no capital, government spending, entry, or exit.

## 2. Household preferences and heterogeneous CES demand

The household maximizes

$$
\int_0^\infty e^{-\rho t}
\left[u(C_t)-\frac{\chi N_t^{1+\varphi}}{1+\varphi}\right]dt,
\qquad \rho,\chi>0,\quad \varphi>0,
$$

The consumption utility function is explicitly CRRA. The balanced-growth baseline uses its log case $\gamma=1$; the reason is derived in Section 9:

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

For household purchases $q^C_{it}$, this demand formula gives $\int p_{it}q^C_{it}\,di=P_tC_t$. When menu services and R&D also use the same composite good, total sales replace $C_t$ by total composite absorption $Y_t$; see Sections 4 and 7. Taste weights enter both demand and the price index. They cannot be added to demand while retaining an unweighted price index.

Taste dynamics have unit mean-reversion speed:

$$
\boxed{da_{it}=(\bar a_{it}-a_{it})\,dt+\sigma_a\,dZ_{it}},
\qquad \sigma_a>0.
$$

$Z_{it}$ is standard Brownian motion for good $i$, independent across goods. The subscript makes explicit that the shocks are idiosyncratic; a common $Z_t$ would instead introduce aggregate uncertainty. The drift is multiplied by $dt$.

The firm does not observe its latent quality $\theta_{it}\equiv\bar a_{it}$. Quality stays fixed between successful innovations and is redrawn when R&D succeeds. The firm infers quality from its own past sales, as described below. The household experiences current tastes but does not reveal $\theta_{it}$ or private information about it to the firm; aggregate prices convey no additional firm-specific signal in the continuum economy.

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

## 4. Revenue-funded menu costs and R&D

### Production and expenditures

Each firm has linear technology $q_{it}=A n_{it}$, $A>0$, and real marginal cost $m_t=w_t/A$. Define $x_{it}=\log(p_{it}/P_t)$. A competitive composite-good sector combines varieties using the same CES technology and weights as household consumption. Its output serves consumption, menu services, and R&D. Let $Y_t$ be total composite absorption. Total variety sales and real revenue are

$$
q_{it}=e^{a_{it}-\epsilon x_{it}}Y_t,
\qquad R_t(x,a)=e^{a+(1-\epsilon)x}Y_t.
$$

Let $Q_t=\exp[M_t/(\epsilon-1)]$ denote the common quality scale. Each price change costs $Q_tK$ units of the composite good, or $P_tQ_tK$ nominally, where $K>0$ is fixed in quality-normalized units. The firm pays it from revenue, reducing its dividend by $P_tQ_tK$ at the adjustment. It is a fixed expense rather than a direct labor requirement or a percentage of instantaneous revenue. R&D effort $k_{it}\geq0$ is measured in quality-normalized composite goods per unit time. Its actual composite-good expenditure is $Q_tk_{it}$, also deducted from revenue. These scaling conventions are required for the stationary balanced-growth version: keeping a fixed cost in unscaled composite units would make its relative burden vanish over time. Neither expense is rebated to the household: the purchased goods are used up in menu services and research.

There is no cash-in-advance or borrowing constraint. Paying from revenue specifies net cash flow, not an additional liquidity constraint on a lump-sum payment. An explicit retained-earnings state would be needed to impose such a constraint.

Real operating profit before these expenses is

$$
\Pi_t(x,a)=\left(e^x-\frac{w_t}{A}\right)e^{a-\epsilon x}Y_t.
$$

Between price adjustments $\dot x_{it}=-\pi_t$. At adjustment the firm chooses a new $x=\xi$ and pays $K$; the action itself changes neither quality nor beliefs.

### Innovation technology and quality draws

The firm chooses an adapted R&D flow $k_{it}\geq0$. Successful innovations arrive according to an observed counting process $H_{it}$ with controlled intensity

$$
\boxed{\lambda(k)=\lambda_0 k^\omega,\qquad \lambda_0>0,\quad 0<\omega<1.}
$$

This technology is increasing and strictly concave for $k>0$, with $\lambda(0)=0$ and the Inada conditions

$$
\lambda'(k)=\lambda_0\omega k^{\omega-1},\qquad
\lim_{k\downarrow0}\lambda'(k)=\infty,\qquad
\lim_{k\to\infty}\lambda'(k)=0.
$$

For any finite, strictly positive value gain from innovation, these conditions give a unique positive, finite optimal R&D flow under the linear expenditure cost. Admissible policies must induce a nonexplosive innovation process and finite expected discounted expenditures.

Let the average market quality be the cross-sectional mean of **log quality**,

$$
M_t=\int_0^1\theta_{it}\,di.
$$

At an innovation time $T$, the firm's quality is replaced by

$$
\boxed{\theta_{iT}=M_{T^-}+\delta_Q+\sigma_Q\varepsilon_{iT},
\qquad \varepsilon_{iT}\sim\mathcal N(0,1),\quad \sigma_Q>0,\quad \delta_Q>0.}
$$

The draws are independent across events and firms and independent of taste Brownian shocks. Equivalently,

$$
d\theta_{it}=(M_{t^-}+\delta_Q+\sigma_Q\varepsilon_{it}-\theta_{it^-})\,dH_{it}.
$$

The mean-reversion process remains $da_{it}=(\theta_{it}-a_{it})dt+\sigma_a dZ_{it}$. Current taste $a_{it}$ is continuous at innovation; its future drift changes. The firm sees the success event but not the new draw. Conditional on that event, its new prior is $\mathcal N(M_{T^-}+\delta_Q,\sigma_Q^2)$, and earlier sales do not inform the independent new draw. The aggregate $M_t$ is public and deterministic in the continuum economy.

The parameter $\delta_Q>0$ is the innovation premium in log quality:

$$
\mathbb E[\theta_{iT}\mid\text{innovation at }T]=M_{T^-}+\delta_Q>M_{T^-}.
$$

An innovation is therefore better than average market log quality in expectation, while the Gaussian component still allows disappointing draws or a decline relative to the firm's own previous quality. The premium is measured relative to average **log** quality, rather than the arithmetic mean of exponentiated quality. Setting $\delta_Q=0$ recovers the mean-centered draw. With the power intensity above, R&D is positive for every strictly positive value gain from innovation; it is zero if that gain is nonpositive.

### Firm objective

With $s=(x,a,b,v)$ denoting the observed pricing and belief state defined below, the firm maximizes

$$
V_t(s)=\sup_{\{\tau_n,\xi_n\},\{k_u\}}
\mathbb E\!\left[
\int_t^\infty D^R_{t,u}[\Pi_u(x_u,a_u)-Q_uk_u]du
-\sum_{n:\tau_n\geq t}D^R_{t,\tau_n}Q_{\tau_n}K
\middle|\mathcal F^i_t\right].
$$

Policies have no finite accumulation of price interventions, finite expected R&D expenditures, and well-defined discounted payoffs. Innovations do not change nominal prices automatically; the firm may choose to pay $Q_tK$ and reprice immediately after a success.

## 5. Sales-based Bayesian learning and optimal pricing

### Information and the Gaussian posterior

The firm continuously observes its own total sales quantity, its price, its innovation-success events, and the aggregate variables $(P_t,Y_t,C_t,w_t,M_t)$. There is no measurement error, rationing, or stockout. CES demand implies

$$
a_{it}=\log q_{it}+\epsilon\log(p_{it}/P_t)-\log Y_t.
$$

Thus sales reveal current $a_{it}$ exactly, but not its latent quality $\theta_{it}$. A price change does not create a taste observation jump: its known effect on sales is removed by this formula. Let $\mathcal F^i_t$ contain this observed history and the firm's initial information. Specify the Gaussian prior **after observing initial sales**:

$$
\theta_{i0}\mid\mathcal F^i_0\sim\mathcal N(b_{i0},v_{i0}),\qquad v_{i0}>0.
$$

Between observed innovation events, let $T$ be the most recent success (or $0$ initially), with initial posterior $(b_{iT},v_{iT})$. Define

$$
Y^a_{i,t;T}=a_{it}-a_{iT}+\int_T^t a_{iu}\,du,
\qquad dY^a_{it}=\theta_{it}\,dt+\sigma_a dZ_{it}.
$$

On this interval quality is constant, and the posterior is

$$
\theta_{it}\mid\mathcal F^i_t\sim\mathcal N(b_{it},v_{it}),
$$

$$
\boxed{v_{it}=\left(v_{iT}^{-1}+(t-T)/\sigma_a^2\right)^{-1},
\qquad
b_{it}=v_{it}\left(\frac{b_{iT}}{v_{iT}}+\frac{Y^a_{i,t;T}}{\sigma_a^2}\right).}
$$

These are the interval-wise linear-Gaussian filtering formulas; see [Kalman–Bucy filtering lecture notes](https://maxim.ece.illinois.edu/teaching/spring19/notes/week14.pdf). With innovation Brownian motion

$$
dI_{it}=\frac{da_{it}-(b_{it}-a_{it})dt}{\sigma_a},
$$

the between-event separated dynamics are

$$
\begin{aligned}
da_{it}&=(b_{it}-a_{it})dt+\sigma_a dI_{it},\\
db_{it}&=\frac{v_{it}}{\sigma_a}dI_{it},\\
dv_{it}&=-\frac{v_{it}^2}{\sigma_a^2}dt.
\end{aligned}
$$

At an observed success,

$$
\boxed{(x,a,b,v)\longmapsto\mathcal R_t(x,a,b,v)
=(x,a,M_t+\delta_Q,\sigma_Q^2).}
$$

Thus R&D renews uncertainty, while sales reduce it between successes. The survival of the innovation clock provides no additional signal about hidden quality: its intensity depends on the firm's observed policy, not directly on the unobserved true quality. If success times were unobserved, the posterior would generally be a mixture rather than a single Gaussian.

The diffusion covariance is $d\langle a_i,b_i\rangle_t=v_{it}dt$. A price change does not alter sales informativeness; R&D changes the underlying quality and the posterior through its observed reset event.

### Firm value and the QVI

The firm's sufficient state is $s=(x,a,b,v)$. Both pricing and R&D policies are adapted to the observed information. Current profits depend on observed $a$, not directly on the unobserved $\theta_{it}$.

The continuation generator is

$$
\boxed{\mathcal L_t V=-\pi_t V_x+(b-a)V_a
-\frac{v^2}{\sigma_a^2}V_v
+\frac{\sigma_a^2}{2}V_{aa}
+vV_{ab}
+\frac{v^2}{2\sigma_a^2}V_{bb}.}
$$

Define

$$
\mathcal M_tV(x,a,b,v)=\sup_\xi V_t(\xi,a,b,v)-Q_tK.
$$

The pricing and R&D equation is

$$
\boxed{\max\left\{
\partial_tV_t+\mathcal L_tV_t+\Pi_t-r_tV_t
+\sup_{k\geq0}\left[-Q_tk+\lambda(k)(V_t(\mathcal R_ts)-V_t(s))\right],
\mathcal M_tV-V_t
\right\}=0.}
$$

Let $G_t(s)=[V_t(\mathcal R_ts)-V_t(s)]/Q_t$ be the innovation gain in quality-normalized units. The optimal continuation R&D policy solves

$$
\sup_{k\geq0}\{-k+\lambda_0 k^\omega G_t(s)\}.
$$

For finite $G_t(s)>0$, the objective is strictly concave, its derivative tends to $+\infty$ at zero and to $-1$ at infinity, and its unique interior optimum satisfies

$$
\lambda_0\omega(k_t^*(s))^{\omega-1}G_t(s)=1.
$$

Therefore

$$
\boxed{k_t^*(s)=
\begin{cases}
[\lambda_0\omega G_t(s)]^{1/(1-\omega)},&G_t(s)>0,\\
0,&G_t(s)\leq0.
\end{cases}}
$$

The optimized R&D contribution to the QVI is

$$
\boxed{\mathcal H(G)=\frac{1-\omega}{\omega}
[\lambda_0\omega\max\{G,0\}]^{1/(1-\omega)}.}
$$

Thus the continuation equation can equivalently use $Q_t\mathcal H(G_t(s))$ in place of the supremum over $k$. For $G>0$, $\lambda(k^*)G=k^*/\omega$ verifies this expression.

The Inada condition guarantees interior R&D **conditional on a positive innovation value gain**. It does not make an unfavorable reset attractive: the model's noisy draw need not dominate the firm's existing quality, so $G\leq0$ remains possible even with $\delta_Q>0$. An assertion that every firm invests strictly positively would require an additional restriction ensuring $G_t(s)>0$ at every relevant state.

The post-success value already includes the option to pay the menu cost immediately, since $V\geq\mathcal M V$. No separate free price reset accompanies innovation.

The firm waits where $V_t>\mathcal M_tV$ and resets according to

$$
x_t^*(a,b,v)\in\arg\max_\xi V_t(\xi,a,b,v).
$$

If a single inaction band exists, its boundaries are $\ell_t(a,b,v)$ and $u_t(a,b,v)$. At a boundary $x=d_t(a,b,v)$, value matching requires

$$
V_t(d_t(a,b,v),a,b,v)=V_t(x_t^*(a,b,v),a,b,v)-Q_tK.
$$

An interior differentiable reset optimum satisfies $V_x=0$. Boundary smooth fit requires separate regularity justification because the diffusion is degenerate.

## 6. Distribution of firms

Let $\mu_t(ds)$ be the unit-mass distribution of $s=(x,a,b,v)$. The pricing policy determines an intervention flow $J_t(ds)$: the mass per unit time leaving pre-adjustment states. Let

$$
\nu_t=\int J_t(ds)
$$

be aggregate adjustment frequency. With a deterministic reset policy, the weak forward equation is, for smooth test functions $h$,

$$
\boxed{
\frac{d}{dt}\int h\,d\mu_t
=\int[\mathcal L_t h+\lambda(k_t^*(s))(h(\mathcal R_ts)-h(s))]\,d\mu_t
+\int\left[h(x_t^*(a,b,v),a,b,v)-h(x,a,b,v)\right]J_t(ds).
}
$$

The intervention flow is the endogenous boundary outflow induced by the optimal stopping policy, with reinjection at reset states. It is not an exogenous Poisson adjustment rate. It may be supported on boundaries and need not have a density with respect to $\mu_t$.

Taking $h=1$ verifies mass conservation. Price interventions leave $(a,b,v)$ unchanged; innovation events reset $(b,v)$. The forward equation includes both the separated diffusion generator and the R&D jump operator above. Aggregation assumes a correctly specified population prior, so conditional on each observed state true quality has conditional distribution $\mathcal N(b,v)$. If an aggregate adjustment wave produces an atom in time, use the corresponding integrated measure equation rather than assuming a finite instantaneous $\nu_t$.

## 7. Aggregation, quality spillovers, and market clearing

Let $\mu_t(ds)$ be the distribution of observed states, with Bayes-consistent conditional quality distributions. The CES price-index identity and production dispersion are

$$
\boxed{\int e^{a+(1-\epsilon)x}\,d\mu_t=1},
\qquad \Delta_t=\int e^{a-\epsilon x}\,d\mu_t.
$$

Define aggregate quality-normalized R&D effort $\mathcal K_t=\int k_t^*(s)d\mu_t$ and innovation frequency $\Lambda_t=\int\lambda(k_t^*(s))d\mu_t$. The frequency of **price** changes is $\nu_t=\int dJ_t$.

Composite-good and labor clearing require

$$
\boxed{Y_t=C_t+Q_t(K\nu_t+\mathcal K_t)},
\qquad
\boxed{N_t=\frac{Y_t}{A}\Delta_t}.
$$

All variety output is allocated to the composite sector, $q_t(x,a)=e^{a-\epsilon x}Y_t=A n_t(x,a)$. Menu services and R&D absorb goods rather than directly adding labor inputs. Nominal revenues sum to $P_tY_t$. Net real dividends satisfy

$$
\frac{D_t}{P_t}=\int\Pi_t\,d\mu_t-Q_t(K\nu_t+\mathcal K_t)
=Y_t-w_tN_t-Q_t(K\nu_t+\mathcal K_t)=C_t-w_tN_t.
$$

This preserves the household budget identity $C_t=w_tN_t+D_t/P_t$. Merely deleting menu expenses from dividends without identifying their resource use would not close the general-equilibrium accounts.

Under correct Bayesian beliefs, average market quality and its evolution are

$$
\boxed{M_t=\int b\,d\mu_t},
\qquad
\boxed{\dot M_t=\int\lambda(k_t^*(s))(M_t+\delta_Q-b)\,d\mu_t}.
$$

The second equation follows because a successful draw has expected log quality $M_t+\delta_Q$ and the firm's expected old log quality is $b$. Equivalently,

$$
\dot M_t=\delta_Q\Lambda_t+\int\lambda(k_t^*(s))(M_t-b)\,d\mu_t.
$$

The first term is the direct contribution of the innovation premium; the second captures selection into R&D. Firms take this aggregate spillover as given.

## 8. Taylor rule with a balanced-growth intercept

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

## 9. Stationary balanced-growth equilibrium

### Quality trend and stationary coordinates

Seek a constant growth rate $G>0$ of mean **log** quality,

$$
M_t=M_0+Gt,\qquad Q_t=e^{M_t/(\epsilon-1)},\qquad
\boxed{g\equiv\dot Q_t/Q_t=G/(\epsilon-1).}
$$

Use stationary firm states

$$
\widetilde x=x-\log Q_t,\qquad z=a-M_t,\qquad
\beta=b-M_t,\qquad v=\operatorname{Var}(\theta\mid\mathcal F).
$$

The exponent $1/(\epsilon-1)$ follows from the CES aggregator: multiplying every taste weight by $e^{M_t}$ multiplies composite output for a fixed physical allocation by $Q_t$. Set

$$
C_t=Q_tc,\qquad Y_t=Q_ty,\qquad w_t=Q_t\widetilde w,
\qquad V_t=Q_t\mathcal V.
$$

The quantities $(c,y,\widetilde w,N)$ and the distribution of $(\widetilde x,z,\beta,v)$ are constant. Physical variety output and hours are stationary, while **quality-adjusted** consumption and output grow at $g$.

### Preference and cost conditions for balanced growth

With the original separable utility and constant $\chi$, labor supply requires

$$
\widetilde w=\chi N^\varphi c^\gamma Q_t^{\gamma-1}.
$$

Therefore constant positive hours and stationary detrended wages require $\gamma=1$ when $g>0$. The balanced-growth calibration uses log utility, which is a CRRA specification already allowed in Section 2. Its labor condition is

$$
\boxed{\widetilde w=\chi N^\varphi c.}
$$

The earlier $\gamma=2$ calibration cannot support this constant-hours balanced-growth path with unchanged separable preferences. Retaining it would require an explicit preference modification, such as scaling the labor-disutility coefficient by $Q_t^{1-\gamma}$; the implemented baseline does not make that modification. This restriction follows directly from labor supply and is consistent with the balanced-growth preference restrictions discussed by [King, Plosser, and Rebelo (1988)](https://doi.org/10.1016/0304-3932(88)90030-X).

As specified in Section 4, menu expenses are $Q_tK$ and research expenditure is $Q_tk$. The fixed economic objects are overhead and effort in quality-normalized units, paid out of revenue. Innovation intensity is $\lambda_0k^\omega$. If instead research were kept in unscaled spending units with the same intensity coefficient, either intensity or its revenue share would trend and this stationary firm problem would not result.

### Detrended learning and innovation

Between success events,

$$
\begin{aligned}
d\widetilde x&=-(\pi^*+g)dt,\\
dz&=(\beta-z-G)dt+\sigma_a dI,\\
d\beta&=-Gdt+(v/\sigma_a)dI,\\
dv&=-v^2/\sigma_a^2\,dt.
\end{aligned}
$$

The drift $-G$ in beliefs reflects that a firm's true quality stays fixed between innovations while the market mean improves. An observed success resets

$$
\boxed{(\widetilde x,z,\beta,v)\mapsto
\mathcal R(\widetilde x,z,\beta,v)
=(\widetilde x,z,\delta_Q,\sigma_Q^2).}
$$

The true quality draw relative to the current mean is $\delta_Q+\sigma_Q\varepsilon$. Current taste and price do not jump as a direct consequence of success. Its belief reset is independent of calendar time, so repeated innovation sustains a stationary learning distribution.

### Stationary firm problem

Normalized operating profits and the continuation generator are

$$
\widetilde\Pi=(e^{\widetilde x}-\widetilde w/A)
 e^{z-\epsilon\widetilde x}y,
$$

$$
\widetilde{\mathcal L}\mathcal V
=-(\pi^*+g)\mathcal V_{\widetilde x}
+(\beta-z-G)\mathcal V_z-G\mathcal V_\beta
-\frac{v^2}{\sigma_a^2}\mathcal V_v
+\frac{\sigma_a^2}{2}\mathcal V_{zz}
+v\mathcal V_{z\beta}
+\frac{v^2}{2\sigma_a^2}\mathcal V_{\beta\beta}.
$$

Because firm value grows at $g$ while the household discounts real payouts at $\rho+\gamma g$, the effective detrended discount rate is

$$
\widetilde r=\rho+(\gamma-1)g=\rho\quad\text{in the log-utility baseline}.
$$

Define $\widetilde{\mathcal M}\mathcal V=\sup_\xi\mathcal V(\xi,z,\beta,v)-K$ and $\mathcal G=\mathcal V(\mathcal R s)-\mathcal V(s)$. The stationary QVI is

$$
\boxed{\max\left\{
\widetilde\Pi+\widetilde{\mathcal L}\mathcal V
-\widetilde r\mathcal V
+\sup_{k\geq0}[-k+\lambda_0k^\omega\mathcal G],
\widetilde{\mathcal M}\mathcal V-\mathcal V
\right\}=0.}
$$

Consequently $k^*=[\lambda_0\omega\max\{\mathcal G,0\}]^{1/(1-\omega)}$. Price reset targets and inaction regions depend only on the stationary states.

### Invariant distribution and equilibrium equations

Let $\widetilde\mu$ be the stationary unit-mass distribution and $J$ the paid price-intervention flow. For suitable test functions $h$,

$$
0=\int[\widetilde{\mathcal L}h+
\lambda(k^*)(h(\mathcal R s)-h(s))]d\widetilde\mu
+\int[h(\xi^*(z,\beta,v),z,\beta,v)-h(s)]dJ.
$$

The equilibrium equations are

$$
\begin{aligned}
1&=\int d\widetilde\mu,\qquad
1=\int e^{z+(1-\epsilon)\widetilde x}d\widetilde\mu,\\
\Delta&=\int e^{z-\epsilon\widetilde x}d\widetilde\mu,\qquad
\nu=\int dJ,\qquad \mathcal K=\int k^*d\widetilde\mu,\\
y&=c+K\nu+\mathcal K,\qquad N=y\Delta/A,\\
\widetilde w&=\chi N^\varphi c,\qquad
\boxed{\int\beta\,d\widetilde\mu=0},\\
\boxed{G}&=\boxed{\int\lambda(k^*)(\delta_Q-\beta)d\widetilde\mu}.
\end{aligned}
$$

The mean-belief condition ensures the detrending trend is the actual market mean. Invariance applied to $h=\beta$ implies the growth-flow equation, so these are not independent extra conditions. Numerically, choose $(c,y,\widetilde w,G)$ to satisfy price-index normalization, resource clearing, labor supply, and zero mean relative belief, while solving the firm QVI and invariant distribution inside each residual evaluation. Use the growth-flow identity as an additional diagnostic.

The taste-mean diagnostic is $\int z\,d\widetilde\mu=-G$. Existence, uniqueness, and positive growth must be checked rather than assumed; a positive premium alone does not prove them.

### Constant price growth and reconstruction

For any normalization $M_0$ and $P_0>0$, reconstruct

$$
\begin{aligned}
M_t&=M_0+Gt,& Q_t&=e^{M_0/(\epsilon-1)}e^{gt},\\
C_t&=Q_tc,&Y_t&=Q_ty,\qquad w_t=Q_t\widetilde w,\\
P_t&=P_0e^{\pi^*t},&
p_{it}&=P_tQ_te^{\widetilde x_{it}}.
\end{aligned}
$$

Thus the stationary distribution is for log nominal prices net of trend $(\pi^*+g)t$. Individual nominal prices remain constant between paid adjustments, but their cross-sectional trend grows at $\pi^*+g$. The quality-adjusted price index grows at $\pi^*$, the real interest rate is $\rho+\gamma g$, and the nominal rate is $\rho+\gamma g+\pi^*$.

## 10. Equilibrium definition and transitions

Given parameters, a Bayes-consistent initial joint distribution of tastes, true qualities, beliefs, and nominal prices, and a Taylor-rule disturbance path, an equilibrium consists of household choices, firm values, pricing and R&D policies, distributions and event flows, market quality, and aggregate prices and quantities such that:

1. CES consumption demand, labor supply, the household budget, Euler equation, and transversality hold.
2. Each firm learns from sales, resets its Gaussian prior at observed R&D successes, and optimizes both pricing and R&D given aggregate paths.
3. True qualities follow the controlled reset law, and the observed-state distribution follows the diffusion, innovation jumps, and price-reset flows.
4. Composite goods, varieties, labor, and the zero-net-supply bond market clear, with menu and R&D expenditures included in resource use.
5. The CES price-index identity, average-quality consistency, and Taylor rule hold at every date.

For transitions solve the firm QVI backward and the distribution equation forward, updating $(C,Y,w,N,M,\pi)$ to satisfy equilibrium. A stationary terminal condition is appropriate only if that equilibrium exists. Innovation arrivals and quality draws remain idiosyncratic; aggregate stochastic shocks would require aggregate state variables and a stochastic household discount factor.

## 11. Economic interpretation and benchmarks

The frictionless relative price is

$$
e^{x_t^{\mathrm{flex}}}=\frac{\epsilon}{\epsilon-1}\frac{w_t}{A}.
$$

Heterogeneous tastes shift demand levels but do not directly shift the frictionless markup under CES demand and linear technology. With menu costs, they affect the value of correcting a price, while general-equilibrium wages can move the frictionless target. Taste uncertainty, learning, R&D incentives, labor supply, and aggregate pricing therefore interact. The latent quality draw changes future demand, while observed current taste determines current sales.

For a useful exact benchmark, shut down R&D, let menu costs tend to zero, and assume stationary tastes with $\int e^{a_i}\,di=1$. Flexible relative prices are identical, and the price-index identity gives

$$
e^{x^{\mathrm{flex}}}=1,\qquad
w=A\frac{\epsilon-1}{\epsilon},\qquad
N=C/A.
$$

Consequently

$$
\boxed{C^{\mathrm{flex}}=
\left[\frac{A^{1+\varphi}(\epsilon-1)}{\chi\epsilon}\right]^{1/(\gamma+\varphi)}}.
$$

Variety quantities still differ: $q_i=e^{a_i} C$. This benchmark reflects the monopoly distortion and requires no markup subsidy.

## 12. Computational plan and validation

1. Use `calibration_bgp.toml`: log CRRA utility, quality-normalized overhead and research, the innovation premium, and the inflation target. Choose stationary grids for the price, taste, and belief gaps and for posterior variance.
2. For candidate $(c,y,\widetilde w,G)$, solve the stationary pricing/R&D QVI with effective discount rate $\rho$ and price-gap drift $-(\pi^*+G/(\epsilon-1))$.
3. Solve the induced invariant distribution, including paid price resets and successful-innovation belief resets. Use zero mean relative belief to determine the quality trend rather than setting quality drift to zero.
4. Solve the four aggregate residuals in Section 9. Verify the resulting growth-flow identity and the CES, labor, resource, and dividend-budget identities.
5. Check filtering covariance, Gaussian prior resets, nonnegative mass, price intervention optimality, R&D optimality, and the stationary Bellman and distribution residuals. The code tests these primitives and a full equilibrium solve.
6. Refine the state grid and time step jointly, expand domains, and reduce numerical price smoothing. A smaller time step on an unchanged interpolation grid is a sensitivity experiment, not by itself a convergence proof.
7. Report detrended aggregates, quality and consumption growth, real and nominal interest rates, quality-adjusted inflation, nominal-price trend, R&D spending, price and innovation frequencies, clipping, and continuous and discrete growth-flow errors.

The model has a working finite-grid stationary solver, with numerical checks reported in Section 16. Existence or uniqueness outside the computed parameterization and a fully converged continuum calibration remain open numerical and economic questions. The earlier level-transition code is retained separately and does not define the balanced-growth equilibrium.

## 13. Literature-based calibration

### Units and scope

Use **one year as one unit of time**. For the active balanced-growth baseline, apply the log-utility and quality-normalized-cost overrides in Sections 9 and 16; the earlier $\gamma=2$ benchmark below remains a literature comparison. The table below provides an initial U.S.-oriented benchmark, not an estimated parameter vector for this model. Values borrowed from related models are distinguished from normalizations and parameters requiring new estimation. In particular, the combination of hidden taste quality, learning, and endogenous quality replacement does not have a ready-made calibration in the sources below.

For a discrete discount factor $\beta_\Delta$ over $\Delta$ years, the exact continuous-time conversion is

$$
\rho=-\log(\beta_\Delta)/\Delta.
$$

Thus quarterly $\beta=0.99$ corresponds to $\rho=0.0402013$ per year. A 2% annual gross price increase corresponds to $\pi^*=\log(1.02)=0.0198026$ per year, since this model defines inflation as $d\log P_t/dt$. Feedback coefficients on interest rates and inflation are unchanged when both rates are converted using the same time unit.

### Parameters with established benchmarks

| Model parameter | Initial benchmark | Literature value and interpretation | Reference |
| --- | --- | --- | --- |
| Discount rate $\rho$ | $0.04$ per year | Annual continuous-time rate in the published menu-cost calibration. | [Golosov and Lucas (2007), Section IV](https://faculty.econ.ucdavis.edu/faculty/kdsalyer/LECTURES/Ecn235a/presentation%20papers/golosov_lucas.pdf) |
| CRRA coefficient $\gamma$ | $2$ | Their risk-aversion calibration; log utility, $\gamma=1$, is an alternative textbook benchmark. | [Golosov and Lucas (2007), Section IV](https://faculty.econ.ucdavis.edu/faculty/kdsalyer/LECTURES/Ecn235a/presentation%20papers/golosov_lucas.pdf); [Galí (2018), calibration slide](https://crei.cat/wp-content/uploads/2016/07/basic_nk_may2018.pdf) |
| CES elasticity $\epsilon$ | $7$ | Menu-cost benchmark implying a flexible-price gross markup $7/6$. Galí uses $9$, useful as a sensitivity case. | [Golosov and Lucas (2007), Section IV](https://faculty.econ.ucdavis.edu/faculty/kdsalyer/LECTURES/Ecn235a/presentation%20papers/golosov_lucas.pdf); [Galí (2018), calibration slide](https://crei.cat/wp-content/uploads/2016/07/basic_nk_may2018.pdf) |
| Inverse Frisch elasticity $\varphi$ | $5$ | Galí's household calibration implies Frisch elasticity $1/5$. This is a preference benchmark, not an estimate for this model. | [Galí (2018), calibration slide](https://crei.cat/wp-content/uploads/2016/07/basic_nk_may2018.pdf) |
| Inflation feedback $\phi_\pi$ | $1.5$ | Standard textbook Taylor-rule coefficient. The source also includes an output response; this model deliberately retains only inflation feedback. | [Galí (2018), calibration slide](https://crei.cat/wp-content/uploads/2016/07/basic_nk_may2018.pdf) |
| Inflation target $\pi^*$ | $\log(1.02)$ per year | A policy benchmark based on a 2% annual target; the official target refers to PCE inflation, whereas $P_t$ here is the model's CES index. | [FOMC (2025), Statement on Longer-Run Goals](https://www.federalreserve.gov/monetarypolicy/monetary-policy-strategy-tools-and-communications-statement-on-longer-run-goals-monetary-policy-strategy-2025.htm) |
| R&D elasticity $\omega$ | $0.5$ | A square-root innovation technology is consistent with the innovation/R&D curvature calibrated in their model. Their equation (8) has research-input exponent $1-\gamma_{\mathrm{A}}$ and they set $\gamma_{\mathrm{A}}=0.5$; hence the corresponding exponent here is $\omega=0.5$. The underlying research input there is skilled labor. | [Acemoglu et al. (2018), equation (8) and Table 1](https://economics.mit.edu/sites/default/files/publications/Innovation,%20Reallocation%20and%20Growth.pdf) |
| Productivity $A$ | $1$ | Units normalization, not a literature estimate. | Model convention |
| Taste mean-reversion speed | $1$ per year | Maintained specification from Section 2, not an externally calibrated value. Conditional on fixed quality, the taste gap has half-life $\log 2\approx0.693$ years. | Model assumption |

With $\omega=0.5$, the R&D policy simplifies to

$$
k_t^*(s)=\left(\frac{\lambda_0\max\{G_t(s),0\}}{2}\right)^2.
$$

### Menu cost and price-adjustment targets

A useful cost target is **aggregate menu expenditure divided by aggregate revenue**, not a fixed cost expressed as an unqualified number:

$$
s_{\mathrm{menu}}=\frac{Q_tK\nu}{Y_t}=\frac{K\nu}{y}.
$$

Golosov and Lucas report approximately **0.5% of revenue** in their baseline, with empirical targets of **21.9% monthly price-change frequency** and **9.5% mean log price increase conditional on an increase**. These are distinct moments, not a single adjustment rate and an unconditional absolute size. [Golosov and Lucas (2007), Table 1 and Section IV](https://faculty.econ.ucdavis.edu/faculty/kdsalyer/LECTURES/Ecn235a/presentation%20papers/golosov_lucas.pdf).

Direct supermarket evidence gives **0.70% of revenue** spent on menu costs. Use $s_{\mathrm{menu}}\in[0.005,0.007]$ as a source-based comparison band, not as a universal estimate across industries. The supermarket evidence concerns multiproduct retailers, while this model has one good per firm. [Levy et al. (1997), abstract and Table IV](https://academic.oup.com/qje/article-abstract/112/3/791/1926906).

Given candidate detrended equilibrium revenue $y=Y/Q$ and annual price-adjustment flow $\nu$, the cost-share target implies

$$
K=s_{\mathrm{menu}}\frac{y}{\nu}.
$$

This is an equilibrium calibration condition: changing $K$ changes both $y$ and $\nu$. Do not set $K=0.005$ merely because the target expenditure share is 0.5%. Jointly fit $K$ and demand-process parameters to price frequencies and sizes, then check the implied cost share. Simulated monthly price-change indicators should match the data's observation interval; $12$ times a monthly fraction, or a Poisson hazard conversion, need not equal the endogenous annual intervention flow when firms can change prices multiple times.

### Learning and innovation parameters requiring model-specific estimation

| Parameter | Calibration status | Suggested identifying information |
| --- | --- | --- |
| Taste volatility $\sigma_a$ | Estimate jointly with pricing and learning parameters; productivity-shock calibrations are not directly transferable. | Sales residuals $a=\log q+\epsilon x-\log Y$, their high-frequency variation, and price-change size and frequency. With continuous noiseless observation, $d[a]_t/dt=\sigma_a^2$. |
| Innovation productivity $\lambda_0$ | No directly transferable level: it depends on the units and definition of R&D spending. | Innovation-success counts conditional on expenditure, and the aggregate R&D/revenue ratio $\mathcal K/Y$. Conditional expected counts over $[0,T]$ are $\lambda_0\int_0^T k_t^\omega dt$. |
| Mean quality premium $\delta_Q$ | Estimate for this demand-quality reset mechanism. | Post-success changes in inferred long-run demand relative to pre-success market-average log quality. |
| Quality-draw dispersion $\sigma_Q$ | Estimate jointly with the latent-quality process. | Dispersion of post-innovation sales trajectories after accounting for taste diffusion, demand mean reversion, and inference error. |
| Initial posterior variance $v_{i0}$ | Initial-information specification, not a universal structural constant. | Pre-sample sales histories or their implied filtering uncertainty. At a newly observed success, the model itself imposes $v=\sigma_Q^2$. |
| Initial posterior mean $b_{i0}$ and market mean $M_0$ | Specify consistently with initial sales and the population quality distribution. | Pre-sample filtering and cross-sectional mean consistency $M_0=\int b\,d\mu_0$. |
| Labor disutility scale $\chi$ | Choose to match hours rather than import a coefficient with different units. | Given a target $N^{\mathrm{tar}}$, impose $\chi=w/[(N^{\mathrm{tar}})^\varphi C^\gamma]$ jointly with equilibrium. A one-third time-endowment target is a possible convention; its mapping to measured annual hours must be stated. |

For context, Acemoglu et al. estimate an innovation step parameter of **0.132** in Table 2. Their innovation changes productivity by an additive amount proportional to market-average productivity (equation (18)), whereas this model redraws a hidden **log demand-quality mean**. Consequently **0.132 is not an estimate of $\delta_Q$ or $\sigma_Q$** and should not be inserted as either. [Acemoglu et al. (2018), equations (8), (18), and Table 2](https://economics.mit.edu/sites/default/files/publications/Innovation,%20Reallocation%20and%20Growth.pdf).

A practical starting set is therefore

$$
(\rho,\gamma,\epsilon,\varphi,\phi_\pi,\pi^*,\omega,A)
=(0.04,2,7,5,1.5,\log(1.02),0.5,1).
$$

This is a **partial calibration**, not a solved equilibrium: $(K,\sigma_a,\lambda_0,\delta_Q,\sigma_Q,\chi)$ and the initial belief distribution still require joint fitting. R&D-success and sales data are needed to separate learning, quality uncertainty, and innovation productivity; price-adjustment moments alone generally do not identify all of them.

### Time-unit conversions for implementation

The unit taste mean-reversion coefficient fixes a physical timescale. It cannot remain numerically one after switching time units while holding the same economy fixed. If $\tau$ measures time in periods of $h$ years, then

$$
da=(h\theta-ha)d\tau+\sqrt h\,\sigma_a dZ_\tau,
\qquad \rho_\tau=h\rho,\quad \pi^*_\tau=h\pi^*.
$$

If R&D spending is also expressed per new period, $k_\tau=hk$, the matching intensity coefficient is

$$
\lambda_{0,\tau}=h^{1-\omega}\lambda_0.
$$

The jump parameters $(\delta_Q,\sigma_Q)$, beliefs $(b,v)$, and lump-sum menu cost $K$ retain their units. Other flow quantities, including revenues and consumption, must be converted consistently; the numerical values of $A$ and $\chi$ depend on the chosen output, labor, and flow conventions.

### References

- King, Robert G., Charles I. Plosser, and Sergio T. Rebelo (1988). “Production, Growth and Business Cycles: I. The Basic Neoclassical Model.” *Journal of Monetary Economics* 21(2–3), 195–232. [DOI](https://doi.org/10.1016/0304-3932(88)90030-X).

- Golosov, Mikhail, and Robert E. Lucas Jr. (2007). “Menu Costs and Phillips Curves.” *Journal of Political Economy* 115(2), 171–199. [Paper](https://faculty.econ.ucdavis.edu/faculty/kdsalyer/LECTURES/Ecn235a/presentation%20papers/golosov_lucas.pdf). [DOI](https://doi.org/10.1086/512625).
- Galí, Jordi (2018). “The Basic New Keynesian Model.” May 2018 lecture slides, calibration based on Galí (2015), *Monetary Policy, Inflation, and the Business Cycle*, second edition. [Author's slides](https://crei.cat/wp-content/uploads/2016/07/basic_nk_may2018.pdf), calibration slide 17/17 (PDF page 23).
- Levy, Daniel, Mark Bergen, Shantanu Dutta, and Robert Venable (1997). “The Magnitude of Menu Costs: Direct Evidence from Large U.S. Supermarket Chains.” *Quarterly Journal of Economics* 112(3), 791–824. [Article](https://academic.oup.com/qje/article-abstract/112/3/791/1926906). [DOI](https://doi.org/10.1162/003355397555352).
- Acemoglu, Daron, Ufuk Akcigit, Harun Alp, Nicholas Bloom, and William R. Kerr (2018). “Innovation, Reallocation, and Growth.” *American Economic Review* 108(11), 3450–3491. [Authors' published paper](https://economics.mit.edu/sites/default/files/publications/Innovation,%20Reallocation%20and%20Growth.pdf). [DOI](https://doi.org/10.1257/aer.20130470).
- Federal Open Market Committee (2025). “Statement on Longer-Run Goals and Monetary Policy Strategy,” amended August 22, 2025. [Official statement](https://www.federalreserve.gov/monetarypolicy/monetary-policy-strategy-tools-and-communications-statement-on-longer-run-goals-monetary-policy-strategy-2025.htm).

## 14. Provisional parameters and balanced-growth overrides

The file [calibration.toml](calibration.toml) records the earlier level-transition benchmark. The active balanced-growth solver uses [calibration_bgp.toml](calibration_bgp.toml), with the overrides in Section 16. Annual units and the literature-backed parameters from Section 13 remain in force. The following additional numbers are **illustrative choices for computation, not estimates or literature calibrations**:

| Parameter or initial condition | Provisional value | Reason for the starting value |
| --- | --- | --- |
| Fixed menu expense $K$ | $0.0015$ composite goods per adjustment | About 0.45% of annual revenue if $Y\approx1/3$ and $\nu\approx1$ per year. Actual equilibrium expenditure shares must be checked. |
| Taste innovation volatility $\sigma_a$ | $0.20$ per square-root year | Conditional on unchanged quality, the stationary taste gap has standard deviation $0.20/\sqrt2\approx0.141$. |
| Innovation scale $\lambda_0$ | $0.50$ | At annual spending $k=0.01$, gives intensity $0.05$ per year when $\omega=0.5$. Spending is in composite-good units, not revenue shares. |
| Innovation premium $\delta_Q$ | $0.05$ log points | A moderate positive quality-mean shift; an initial sensitivity range is $0.02$–$0.10$. |
| Quality-draw standard deviation $\sigma_Q$ | $0.20$ log points | Allows both improvements and disappointments around the shifted market mean. |
| Labor utility scale $\chi$ | $(6/7)3^7=1874.57142857$ | Matches $C=N=1/3$ in the no-R&D, flexible-price normalization with $(A,\gamma,\varphi,\epsilon)=(1,2,5,7)$. |
| Initial mean observed taste | $-0.03$ | A modest initial log-taste level; normalization is not reimposed every period. |
| Initial standard deviation of observed taste | $0.12$ | Initial cross-sectional taste dispersion, approximated by normal quadrature. |
| Initial posterior mean $b_{i0}$ | $-0.03$ for every firm | A common starting belief; its dispersion evolves through sales and innovation. |
| Initial posterior variance $v_{i0}$ | $0.04$ | Matches a fresh innovation prior variance $\sigma_Q^2$. |
| Initial log nominal price | $0$ for every firm | Nominal-unit choice; the initial CES price level is determined in equilibrium. |
| Monetary disturbance $u_t^m$ | $0$ | Baseline policy has no exogenous disturbance. |

The initial information structure is $a_{i0}\sim\mathcal N(-0.03,0.12^2)$ and $\theta_{i0}\mid a_{i0}\sim\mathcal N(-0.03,0.04)$. Conditional on observed $a_{i0}$, quality is drawn independently from that prior. This is Bayes consistent but is not the stationary joint distribution of tastes and qualities. The market mean starts at $M_0=-0.03$. Brownian shocks and innovation marks are independent as specified above. No extra mean-reversion, productivity-shock, entry, or monetary-shock parameters are implicit in the solver.

The initial unconditional mean taste weight is approximately $\exp(-0.03+0.12^2/2)=0.9775$. The price-index identity uses the actual distribution, so this does not require setting that mean to one. Numerical settings are specified separately in `[numerical]`. The BGP solver ignores the legacy transition horizon and terminal-consumption settings; `terminal_iterations` and `terminal_tolerance` are reused as stationary firm iteration controls.

## 15. Legacy transition code

The earlier level-transition implementation is retained in `src/DemandUncertainty.jl` and can be run with `scripts/solve_transition.jl calibration.toml`. It uses the earlier unscaled cost conventions and is not the stationary growth solver. Its former flat-consumption tail is not imposed on the balanced-growth equilibrium. The active implementation is described next.

## 16. Julia stationary balanced-growth solver

Run the active long-run solver with:

```sh
julia --project=. test/runtests.jl
julia --project=. scripts/solve_bgp.jl calibration_bgp.toml
```

The balanced-growth calibration changes $\gamma$ from 2 to **1**, and sets $\chi=(6/7)3^6=624.857142857$ to preserve the no-R&D flexible-price reference $c=N=1/3$. Other economic parameter levels remain as in Section 14. Menu cost $K$ and R&D effort $k$ are now explicitly in quality-normalized units. The initial distribution from the transition calibration is not an equilibrium input: the BGP solver computes an invariant distribution. Its variance bound only helps define the numerical grid. The absolute quality and price levels are arbitrary normalizations, not extra calibration parameters.

[BalancedGrowth.jl](src/BalancedGrowth.jl) solves the stationary firm Bellman equation by sparse Howard policy iteration and solves the induced invariant-distribution linear system. A four-dimensional outer root search determines $(c,y,\widetilde w,G)$. There is no terminal horizon, fixed terminal consumption, or frozen-quality tail. The implementation uses the same positive quadrature, belief covariance, price smoothing, and interpolation machinery as the transition code.

For a time step $h$ without innovation, the exact conditional means in detrended coordinates are

$$
\mathbb E[z']=e^{-h}z+(1-e^{-h})\beta-Gh,
\qquad \mathbb E[\beta']=\beta-Gh.
$$

Conditional variances and covariance follow the same Gaussian filter, using the continuous limit at $v=0$. In particular $v'=(v^{-1}+h/\sigma_a^2)^{-1}$, $\operatorname{Var}(\beta')=v-v'$, $\operatorname{Var}(z')=v(1-e^{-h})^2+\sigma_a^2(1-e^{-2h})/2$, and $\operatorname{Cov}(z',\beta')=v(1-e^{-h})$. A success at step end resets $(\beta',v')=(\delta_Q,\sigma_Q^2)$. Drift in the detrended relative price is $-(\pi^*+g)$; the flexible-price grid center is constant. The normalized discount rate is $\rho$ in the log baseline.

The invariant distribution is sampled just before the discrete pricing decision. Flow aggregation uses the distribution after that decision. These distributions coincide in the continuous-time limit. Because successes are placed at step end, the exact discrete mean-belief balance differs by $O(h)$ from the continuous growth-flow equation; the code reports both errors. Boundary clamping can also distort these identities, and the diagnostics must be small before interpreting the growth estimate.

Outputs in `results_bgp/` include `balanced_growth.toml`, `distribution.csv`, and an aggregate root checkpoint. The CSV reports price gaps relative to the flexible-price center, taste gaps $z$, belief gaps $\beta$, posterior variances, stationary masses, R&D effort, and pricing probabilities. The TOML reports equilibrium, Bellman, mass, and invariant-distribution residuals, boundary clipping, growth-flow errors, and all real and nominal growth rates. A root on a coarse finite grid is an approximation to the balanced-growth equilibrium, not a validated empirical calibration. Time-step, grid, domain, and price-temperature sensitivity remain necessary.


### Earlier development-grid equilibrium and sensitivity

The earlier development grid had $(n_x,n_z,n_\beta,n_v)=(7,11,13,4)$, taste bounds $[-1.5,1.5]$, belief bounds $[-1.2,1.2]$, and a quarter-year step. Both runs below solve the same provisional economic parameterization with log utility. These are finite-grid equilibria; they are **not an empirically estimated or numerically converged continuum calibration**.

| Quantity | Default $h=0.25$ | Time-step sensitivity $h=0.125$, same grid |
| --- | --- | --- |
| Mean log-quality drift $G$ per year | $0.0054570$ | $0.0063969$ |
| Quality-adjusted consumption growth $g=G/6$ | $0.0009095$ (about 0.091%/year) | $0.0010661$ (about 0.107%/year) |
| CES log inflation $\pi^*$ | $\log(1.02)$ | $\log(1.02)$ |
| Log nominal-price trend $\pi^*+g$ | $0.0207121$ | $0.0208688$ |
| Nominal interest rate $\rho+g+\pi^*$ | $0.0607121$ | $0.0608688$ |
| Detrended consumption $c$ | $0.3356781$ | $0.3365591$ |
| Hours $N$ | $0.3336719$ | $0.3337175$ |
| Maximum aggregate residual | $1.36\times10^{-7}$ | $3.21\times10^{-6}$ |
| Invariant-distribution residual (L1) | $2.9\times10^{-16}$ | $3.1\times10^{-16}$ |
| Probability of a clamped transition per step | $0.00315$ | $0.00604$ |
| Absolute continuous growth-flow discrepancy | $9.5\times10^{-6}$ | $3.0\times10^{-5}$ |

The material change in the growth estimate under time-step refinement means that more joint state/time refinement is needed before treating the growth magnitude as precise. The default invariant allocation nevertheless has positive constant growth and constant inflation, with the correct stationary normalization and market-clearing conditions. Quality-adjusted growth and physical production growth are different: physical output and hours are stationary in this equilibrium.

The 77 implementation tests pass, including an end-to-end stationary BGP solve, exact Gaussian moment checks, mass conservation, backward/forward accounting, and the growth-rate transformations. The default calibration has not been fitted to the price-change frequency or menu-expenditure targets in Section 13.

## 17. Full four-dimensional grid and Julia runtime

The active configuration now sets `nx = na = nb = nv = 51`: **6,765,201 states**. Scripts invoke juliaup's default `julia`, currently **Julia 1.13.0**; this preference is recorded in `AGENTS.md`. The project minimum is Julia 1.13.

`src/LargeBGP.jl` factors the observation transition into price and quality/belief/variance operators and uses the exact diagonal-plus-rank-one representation of each price-softmax block. Matrix-free BiCGSTAB performs policy evaluation and solves the invariant-distribution equation. This retains all 51 nodes per dimension and the original discrete-time model. Tests compare the factorized Bellman values, R&D policies, probability flows, and aggregate residuals against the full sparse small-grid implementation.

The requested aggregate report is `bgp_solution_51.csv`, also saved as `results_bgp/solution.csv`. It includes actual grid dimensions, state count, Julia version, growth rates, allocations, and residual diagnostics. The backend does not automatically write a multi-gigabyte state-level distribution CSV. Run the full solve with `JULIA_NUM_THREADS=16 julia --project=. scripts/solve.jl calibration_bgp.toml`.

### Previous wide-price-grid 51-by-51-by-51-by-51 solution

The original full-grid run used juliaup's default **Julia 1.13.0**, 16 threads, and all **6,765,201 states**, with price gaps spanning $[-0.45,0.45]$. The aggregate root converged in four iterations. The original output directory has been removed; the historical values below and current price-grid comparison report retain its summary. The tighter-grid solution below supersedes this table.

| Quantity | Full-grid solution |
| --- | --- |
| Detrended consumption $c$ | $0.3334239447$ |
| Detrended absorption $y$ | $0.3351697350$ |
| Detrended real wage $\widetilde w$ | $0.8620451444$ |
| Hours $N$ | $0.3336956568$ |
| Mean log-quality drift $G$ per year | $0.00308816895$ |
| Quality-adjusted consumption log growth $g=G/6$ | $0.000514694825$ per year |
| Annual consumption growth, $100(e^g-1)$ | $0.05148273\%$ |
| Annual CES inflation, $100(e^{\pi^*}-1)$ | $2.00000000\%$ |
| Annual nominal-price trend, $100(e^{\pi^*+g}-1)$ | $2.05251238\%$ |
| Nominal interest rate, continuous per-year rate | $0.0603173221$ |
| Maximum aggregate equilibrium residual | $6.56\times10^{-7}$ |
| Bellman residual | $7.78\times10^{-11}$ |
| Invariant-distribution residual, L1 | $6.22\times10^{-12}$ |
| Transition clipping probability per step | $7.65\times10^{-6}$ |

The CSV stores raw continuous growth rates as well as explicitly named annual percentage-growth columns. The columns `C`, `Y`, and `w` are detrended levels; `N` is stationary hours. The older aliases `consumption` and `quality` are growth rates, not aggregate levels. The quarter-year step and price smoothing remain unchanged, so this is the converged **finite-grid** BGP for the provisional calibration, not an empirical estimate or a proof of continuous-time convergence. Its lower consumption-growth rate than the earlier development-grid values illustrates the importance of the state-grid refinement.

All 105 implementation tests pass, including the tensor/direct operator comparison, an end-to-end matrix-free equilibrium solve, exact 51-node dimensions, and CSV metadata and growth-unit checks. The exported full-grid result was additionally checked against resource clearing, household labor supply, growth accounting, the interest-rate identity, and the configured equilibrium tolerance.

### Continuous-time firm simulation

`scripts/plot_bgp.jl` simulates the filtered state with a SciML `SDEProblem`:
\[
 d\widetilde x=-(\pi+g)dt,\quad
 dz=(\beta-z-G)dt+\sigma_a dI,\quad
 d\beta=-Gdt+(v/\sigma_a)dI,\quad
 dv=-v^2/\sigma_a^2dt.
\]
A single Brownian innovation drives both stochastic coordinates. `RKMilCommute`
selects adaptive time steps; the equilibrium solver's time step is not imposed on
simulation. Multilinear `Interpolations.jl` policies supply R&D, the price
intervention surplus, and reset prices. The price trigger uses the unsmoothed
continuation comparison underlying the finite-grid pricing policy. A continuous
callback locates its upward zero crossing and executes the price impulse.

A further state integrates the innovation intensity. When it reaches an
independent unit-exponential threshold, a callback resets the posterior mean to
\(\delta_Q\) (the calibrated innovation premium) and variance to
\(\sigma_Q^2\), while leaving taste continuous. The price policy is immediately
reconsidered. Menu payments are deducted as cash jumps, rather than converted
into a flow using the adaptive step size. Policies use flat extrapolation outside
the solved grid without clipping the simulated states. Diagnostics report such
observations. These continuous-time paths use policies estimated from the
finite-step equilibrium approximation; their invariant law need not coincide
exactly with its reported finite-grid marginals.


### Recomputed equilibrium with an efficient price grid

The active calibration uses **51 price nodes on $[-0.075,0.075]$**, measured
relative to the flexible-price grid center. The three other state grids remain
51 nodes each; economic parameters, $h=0.25$, and the pricing temperature
$10^{-5}$ are unchanged. The price spacing falls from **0.018 to 0.003**.
The old equilibrium placed less than $10^{-16}$ mass outside this new range,
and every old grid-state reset target lay between 0.018 and 0.036. The new
bounds therefore retain a margin around the observed support and reset targets.

Values were interpolated and mass was conservatively transferred only to form
an initial guess. The firm policies, invariant distribution, and general
equilibrium were then re-solved. The aggregate root converged in two iterations.
The updated report is [bgp_solution_51.csv](bgp_solution_51.csv), also saved in
`results_bgp/solution.csv`.

| Quantity | Recomputed tighter-grid equilibrium |
| --- | --- |
| Price nodes with stationary mass above $10^{-6}$ | 30 of 51 (previously 6) |
| Effective price nodes, $1/\sum_j\mu_{x,j}^2$ | 20.1685 |
| Mass at the two price endpoints | $3.92\times10^{-36}$ |
| Probability of price-drift clipping per step | $1.33\times10^{-42}$ |
| Detrended consumption $c$ | 0.3333411848 |
| Detrended absorption $y$ | 0.3350699528 |
| Detrended real wage $\widetilde w$ | 0.8612359650 |
| Hours $N$ | 0.3336496018 |
| Mean log-quality drift $G$ per year | 0.003097437617 |
| Annual consumption growth, $100(e^{G/6}-1)$ | 0.05163729% |
| Annual CES inflation | 2.00000000% |
| Price-adjustment frequency per year | 0.3572114791 |
| Innovation frequency per year | 0.0120697892 |
| Maximum equilibrium residual | $1.41\times10^{-6}$ |
| Bellman residual | $2.38\times10^{-14}$ |
| Invariant-distribution residual, L1 | $1.58\times10^{-11}$ |
| Total transition clipping probability per step | $7.63\times10^{-6}$ |

The remaining transition clipping comes from the quality-state domain; price
truncation is negligible at the recomputed equilibrium. All 105 existing solver
checks and five new price-transfer checks pass. Updated marginal and adaptive
single-firm figures are in `results_bgp/figures/`. The before/after figure is
`price_grid_comparison.png`, with numerical changes in
`results_bgp/price_grid_comparison.csv`. This improves price resolution while
holding the other numerical approximations fixed; it does not establish joint
state-grid, time-step, or zero-temperature convergence.

For the additional firm-level plots, normalize $P_0=1$ and $M_0=0$ (hence
$Q_0=1$). Reconstruct $P_t=e^{\pi^*t}$,
$p_{it}=e^{\widetilde x_{it}+(\pi^*+g)t}$,
$a_{it}=z_{it}+Gt$, and the demand multiplier $e^{a_{it}}$.
The firm's own initial price and taste retain their stationary draws.
`results_bgp/figures/firm_levels.png` shows nominal and relative prices,
undetrended taste and beliefs, and the taste multiplier; a PDF is also saved.
