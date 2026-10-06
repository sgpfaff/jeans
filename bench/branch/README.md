# Branch selection: ground truth and the measured rates

The numbers quoted in `FORK.md` and in the `jeans.fast.branch` docstring come
from here.

The point of this directory is that the reference must not share the failure
mode it is testing. A denser continuation is still a continuation: if it
crosses a fold it is wrong in the same way the shipped one is. So nothing here
continues anything.

| file | what it is |
|------|------------|
| `fwd.py` | The reduced forward map `(u1, Lam) -> (R, mu)` for a Miyamoto-Nagai disc, written independently of `jeans.fast`. Explicit: one RK4 integration, no root-find. Agrees with the solver's own converged answers to 9.5e-12 over 399 random cases. |
| `enumerate_roots.py` | Exhaustive root enumeration. Evaluates the forward map on a dense grid, takes every cell where both residuals change sign, polishes with Newton and deduplicates. Finds every root in the box, including ones no continuation path reaches. |
| `bigvalid.py` | The bracketed solver against that enumeration, with the physical sheet identified properly -- by walking each root down to `u1 -> 0` and requiring `det > 0` throughout, not by the `u1 < 22.544` shortcut, which is sufficient but not necessary. |
| `campaign.py` | Draws from two priors and runs every method plus both screens on each. `wide` is the stream-inference prior and measures the operational rate; `fold` is conditioned toward deep discs and large `R` and exists only to put enough fold events in the sample for usable error bars. |
| `analyse.py` | Classifies each draw against ground truth and prints Wilson intervals. |

## Running it

```
python bigvalid.py                 # ~4 min on 48 cores
python campaign.py wide 4000
python campaign.py fold 4000
python analyse.py
```

`bigvalid.py` is the one to run if you only run one.

## Results as measured

`bigvalid.py`, 300 targets: 236 agree to a worst relative difference of
3.9e-12, 64 agree there is no root, 0 disagreements, 0 missed, 0 phantom, and
no target had more than one root on the physical sheet.

`campaign.py` + `analyse.py`, 4000 draws per prior: see the table in
`FORK.md`. Timings were taken on a shared machine and should be re-measured
before being quoted anywhere that matters; the ratios are stable, the absolute
numbers are not.
