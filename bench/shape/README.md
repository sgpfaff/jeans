# How far does the linear shape sector actually go?

`solver2d` linearises the L>0 sector about the spherical background. Its
docstring used to quote "max |phi_2| ... below 0.272 over 1080
configurations" as the validated range, with no statement of what exceeding
it costs. Two things were wrong with that: 35% of a realistic prior exceeds
it, and exceeding it turns out to cost almost nothing.

The reference is the package's own relaxation. It builds its source as the
exact angular integral of `exp(-phi_b - sum_L phi_L Z_L)` and Newton-iterates
on it, so it makes no linearity assumption and is a valid independent check of
exactly this question.

| file | what it does |
|------|--------------|
| `psi_scan.py` | how large `max abs(phi_2)` gets over a realistic prior, by `q0` |
| `psi_err.py` | the error of linear response against the package's nonlinear relaxation, binned in `max abs(phi_2)` |
| `psi_floor.py` | a resolution ladder, to tell a real error from the comparison floor |

```
python psi_scan.py      # ~2 min on 24 cores
python psi_err.py       # ~2 min on 28 cores
python psi_floor.py     # slower; the package at r_grid=1600 is not quick
```
