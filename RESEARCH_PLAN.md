# Research plan: what jeanie is for

Drafted 2026-10-06 after a day that ruled out the obvious route. Living
document — update it as things land, and record what killed an idea, not just
that it died.

## The thesis

The field identifies SIDM observables in one of two regimes, and both are
crippled in the same way:

* **analytic arguments**, which marginalise nothing and usually drop baryons
  entirely;
* **full cosmological simulations**, which include baryons but afford tens of
  haloes, so the summary statistic must be hand-picked *before* looking, and
  the baryon parameters are frozen into whichever subgrid model was run.

Neither can ask *which observable survives marginalising over the baryons*,
because neither can sample baryons. jeanie sits in the Goldilocks zone: fast
enough for 10^5–10^6 haloes, differentiable, with baryons in the equations
rather than in a subgrid recipe. That question is available here and nowhere
else, and it is exactly the question on which every hand-picked metric failed
(see "What was ruled out"). We treat that as evidence that hand-picking is the
wrong method, not that the metrics were unlucky.

## Two prongs

### Prong 1 — the boundary machine (unblocked, start now)

> Is there an observed galaxy whose dark matter core is too large, or too
> underdense, for thermalised dark matter to produce at **any** cross-section,
> **any** age, **any** interaction strength?

The existence criterion is `1/3 < R < R_fold(mu; shape)` with
`R = M1/(4 pi r1^3 rho1)` and `mu = Md/(4 pi r1^3 rho1)` (`branch.py:24`).
Two load-bearing properties, both verified in source:

1. **It is cross-section free.** `cross_section()` (`jeans/classes.py:1162`) is
   pure post-processing: `sigma/m = Nm * conv / (rho1 * vrel * t_age)`. So
   `d log(floor)/d log t_age = 0` exactly, while every published
   isothermal-Jeans `sigma/m` carries `d log(sigma/m)/d log t_age = -1`. Nobody
   marginalises halo age. An existence statement is immune to the largest
   unmarginalised systematic in the literature.
2. **It is a curve, not a grid.** All of (M200, c, Md, r1) enter only through
   two dimensionless ratios, plus the baryon *shape*. Five parameters reduce to
   two. This is the dimensional reduction the discovery search hopes to find,
   obtained analytically — and therefore an existence proof that the problem
   admits such reductions.

The fold is physics, not numerics: it is the classical isothermal (Antonov)
instability, turning points spaced `exp(2 pi / sqrt 7) = 10.7483` against
measured 10.86 / 10.76 / 10.75.

**Three statements come out, with decreasing robustness.**

| | statement | robustness |
|---|---|---|
| existence | a halo outside the reachable set cannot be made by thermalised DM at any parameters | parameter-free; strongest |
| **occupancy** | **a pile-up against `R_fold` is positive evidence for thermalisation — CDM has no reason to respect that locus** | population-level; this is the discrimination |
| position | where a halo sits inside the region encodes r1, hence sigma/m | inherits the t_age and Nm degeneracy; weakest |

Limitation to state up front: the (R, mu) plane is intrinsically an SIDM
construction, since R is defined at r1 and CDM has no r1. The boundary test
discriminates thermalised-vs-not, not SIDM-vs-CDM. Discrimination against CDM
comes through *occupancy*, not through existence.

**First deliverables.**

1. **The `R_fold(mu)` curve — DONE 2026-10-07.** `bench/boundary/`. The
   dimensionless claim is verified, not assumed: `R_fold` is identical to 8
   decimal places across `r1` = 3-40 kpc and `rho1` = 2.5e5-1e7, so it really
   does depend only on `(mu, shape)`. It runs from 1.2615 at `mu=0` (matching
   `universal.R_MAX` to 6e-7) to ~10 at `mu=60`.

   Baryon shape matters more than expected and was nearly dismissed: `R_fold`
   spans 0.42x to 1.18x the thin-disc value, and moves in OPPOSITE directions
   — extended baryons raise the boundary, a compact central concentration
   lowers it (deepens the well, destabilises). So the boundary is a family
   indexed by the stellar distribution. That is an observable, so it is a
   measurement requirement rather than a modelling problem.

   A halo is a TRAJECTORY in `(mu, R)`, not a point, because `r1` is not
   observable: it is reachable iff the trajectory enters `1/3 < R < R_fold`
   somewhere. Four test haloes all do, which is the expected null.

   Still to do here: the linear shell kernel `K(s)` by autodiff, replacing the
   `eps=1e-5` finite difference in `_det` (`branch.py:244`).
2. **External analytic check — DONE 2026-10-07, passes.** The canonical
   turning point of the same march sits at `rho_c/rho_edge = 14.0420` against
   the classical Bonnor-Ebert value `14.04`: four significant figures, nothing
   fitted. The march IS the classical isothermal sphere, so `R_fold` is a real
   turning point and not a convergence artifact. `bench/boundary/`,
   `tests/test_boundary_is_classical.py`.

   The originally planned check was wrong and would have "failed" for the
   wrong reason: it asked for Antonov's microcanonical `708.61` AT OUR FOLD.
   `R = rho_bar/(3 rho_edge)` is its own functional and folds at a different
   turning point of the same sequence — we get `290.47`. Which turning point R
   folds at is a separate question from whether the sequence is classical.
3. Count simulated SIDM haloes falling below their own predicted floor.
   Expect zero, plus a pile-up. More than ~5% below → it is not a floor.

### Prong 2 — the discovery search (gated)

Do not propose observables and test them. Sample the space and let the data
say which functionals carry baryon-robust information.

* Generate ~10^5–10^6 haloes sampling the cross-section **and every nuisance**,
  so baryon marginalisation is built into the training distribution rather than
  applied afterwards.
* Observable is the full discretised `rho(r, theta)` field, not a summary.
* Learn the compression — a regressor for sigma/m and a classifier for
  SIDM-vs-CDM — so the network must find nuisance-robust features.
* **Interpret it**: saliency over (r, theta), symbolic regression onto closed
  forms, deliberately restricted architectures. "A network uses this" must
  become "the informative quantity is X" or it is not physics.
* The literature's statistics (shape, core radius, central density, the
  sphericalization radius d1) are a **baseline to beat**, not the hypothesis
  space. Recovering d1 validates the method; beating it is the result.

**Nothing is pre-excluded, including shape.** Pruning the hypothesis space
using conclusions from a restricted search is the error this whole programme
exists to avoid.

## The gate: a symmetric collisionless CDM solve

Prong 2 cannot run until CDM's shape is *derived* rather than *imposed*.

Today the model couples baryons to shape only through the isothermal solve, so
SIDM's shape is solved for and CDM's is stipulated constant. A hand test
against that straw man gave an inflated number that one extra nuisance
parameter destroyed (chi2 203.5 → 15.2, a 90–94% collapse). **In an automated
search the same asymmetry is far more dangerous**: the search optimises against
whatever weakness the alternative has, so a network would latch onto shape as
the dominant discriminant and report it as a discovery — correctly, about the
straw man — and it would be much harder to catch than a number.

The CDM halo must therefore **sphericalize in the vicinity of the baryons**,
as real haloes do (c/a ~ 0.5 dark-matter-only to ~0.8 with hydro).

**Planned approach.** For an isotropic distribution function `f(E)` the density
is a function of the relative potential alone, so isodensity surfaces follow
isopotential surfaces — the *same* structural statement as the isothermal case,
with a different `rho(Psi)`. That gives a symmetric treatment for free:

* SIDM: `rho = rho_0 exp(-Psi / sigma_0^2)` — the existing solver.
* CDM: `rho = F(Psi)`, with `F` calibrated parametrically from the spherical
  no-baryon solution, then evaluated on the *total* `Psi` including baryons.

Baryonic rounding and contraction both then emerge rather than being imposed,
and `solver2d`'s validated quadrupole linear-response machinery is reused with
`exp(-Psi)` generalised to `F(Psi)`. The self-consistency loop is unchanged.

### Result (2026-10-06): shape cannot discriminate at first order

Built and measured, single pass with Psi_DM spherical. Two findings:

**If rho = F(Psi) with F monotonic, the halo SHAPE is independent of F.**
Isodensity surfaces are isopotential surfaces whatever F is, so collisionless
CDM and isothermal SIDM in the same total potential have *identical* axis
ratios -- the curves coincide exactly, and F cancels out. Shape differences are
therefore SECOND order, arising only through the back-reaction of different
radial profiles on Psi_DM. This explains, in one line, the saturation
(dq/dlog10(sigma/m) ~ 0.05), the 1.3% estimation share, and why the
shape-gradient signature died under one nuisance parameter.

**CDM's own baryon-induced gradient exceeds the "SIDM signature".** Max
|dq/dlnr| = 0.033 / 0.083 / 0.131 for Md = 1 / 3 / 6 x 10^10, against the 0.051
previously reported as an SIDM signature. Derived inside the model, not
inferred from five EAGLE haloes.

### Consequence: anisotropy is the SIGNAL, not a nuisance

Previously listed here as a limitation to patch for fairness. It is the
physical origin of any shape difference at all. SIDM thermalises, so it is
isotropic and its density follows the isopotentials exactly. CDM runs
beta ~ 0.2-0.3, and an anisotropic DF does NOT give rho = F(Psi), so its shape
departs from the isopotential surfaces. **The beta closure is where the
SIDM-vs-CDM shape discriminant lives**, and the discovery search should range
over observables sensitive to anisotropy rather than to the density profile's
angular structure alone.

Still open: real haloes are also triaxial, which an axisymmetric model cannot
represent, so the loss of triaxiality that drives b/a -> 1 in simulations is
out of reach here regardless of closure.

## What was ruled out (all measured, 2026-10-06)

1. Core shape saturates: `dq/dlog10(sigma/m) ~ 0.05`, independent of baryon
   fraction from 0 to 0.81 inside 3 kpc. The paper has the analytic law,
   `k = sqrt(2)/5`. Shape is a detection statistic, not an estimator.
2. The optimal *linear* estimator is density-profile **curvature** (69% curv +
   22% higher; gradient 2.7%, normalisation 6%) — but that search could not
   express feature-position statistics like d1 or r1, so it is a restricted
   answer, not the answer.
3. No universal statistic: over 42 points, an SVD of the optimal compression
   directions gives mode 1 only 40.1% of the variance, and the 10th percentile
   of alignment is 0.199.
4. Intrinsic halo-to-halo scatter dominates measurement error by ~7x. Every
   forecast using instrumental errors is meaningless for a population.
5. `sigma(log10 sigma/m)` spans 0.011–0.13 dex across halo parameters. A
   single-halo forecast number means nothing without saying which halo.
6. Discrimination gets *harder* at larger sigma/m (33 sigma at 0.1 → 21 at
   1.0): a big core is well mimicked by low-concentration CDM.
7. The shape-gradient signature was an artifact of CDM's rigidity (see the
   gate).
8. There is a reachability boundary. **The `sigma/m <~ 1.2` "ceiling" reported
   on the day was a bisection convergence failure, not the fold — retracted.**
9. At 10^12–10^13 with baryons, spherically-averaged SIDM and CDM profiles are
   reported as nearly identical (arXiv:2511.10765), and SIDM can be *denser*
   than CDM at MW mass (Robles+19).

## Metric principles (earned the hard way)

* **Name the estimand before computing.** Local Fisher about sigma/m and
  profiled KL to CDM are different questions with different answers.
* **The alternative must be as flexible as nature.** Any discriminating power
  traceable to the alternative's rigidity is fake. The diagnostic is to add
  nuisance freedom until the answer stops moving. On the shape test it moved
  90% on the first try.
* **The noise must be the noise that limits you** — intrinsic scatter, not
  instrumental error, whenever the inference is over a population.
* **The metric must be invariant to non-physics**: reparametrisation (q vs
  ln q), radial window, units, choice of sampling radii.
* **A model cannot audit its own omissions.** Approximations have parameters
  to vary; absences do not. External input is needed for absences only, and
  observations beat simulations for it where available.
* **Simulations are an existence proof for omitted phenomena, not an arbiter of
  truth.** "CDM haloes have shape gradients" is robust to subgrid choices; any
  quantitative profile is not.

## Stop doing

* Producing `sigma/m` point estimates. 0.48 dex inter-method systematic
  (Yang+23) against a 0.12 dex state of the art.
* Quoting the 0.15 ln q scatter as measured — it is 5 haloes at 10^13–10^14.
* Quoting the retracted `sigma/m < 1.2` ceiling as a bound.

## Calibration: what to beat

| probe | sigma/m (cm^2/g) | v (km/s) | precision |
|---|---|---|---|
| Andrade+22, 8 clusters | 0.082 (+0.027/−0.021) | ~1458 | ±0.12 dex on the mean, 0.27 dex per cluster |
| Sagunski+21 clusters | 0.19 ± 0.09 | ~1900 | |
| Sagunski+21 groups | 0.5 ± 0.2 | ~1150 | |

Starkman+26 (arXiv:2609.40057): extragalactic stream convexity needs
850 / 3400 / 7700 / 21000 streams for 1/2/3/5 sigma; Euclid forecasts ~2300.

## Standing constraint

`/geir_data/scr/gabrielspace/jeans` is the upstream repo and **must stay
clean**. All work happens in this fork.
