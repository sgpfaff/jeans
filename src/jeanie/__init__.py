"""jeanie: fast, differentiable solvers for the isothermal Jeans SIDM model.

A reimplementation of the model in arXiv:2511.10765 (Bautista, Robertson,
Sagunski, Smith-Orlik & Tulin) by a different numerical route, plus two
results about the model that the implementation made visible: an exact
existence criterion with baryons, and a branch criterion for the matching
problem, which is multi-valued.

The package depends on nothing from `jeans` at runtime; the test suite uses
it as the reference to validate against.
"""
