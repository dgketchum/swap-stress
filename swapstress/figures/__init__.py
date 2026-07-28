"""Figures and tables for the Scientific Data descriptor.

Modules here still carry their pre-refactor names; Phase 6 of
``notes/refactor_plan.md`` maps them onto Figs 1-6.
"""

# Shared axis labels. Lived in retention_curve/__init__.py before the package
# was assembled; only figure code has ever used them.
PARAM_SYMBOLS = {
    "theta_r": r"$\theta_r$",
    "theta_s": r"$\theta_s$",
    "alpha": r"$\alpha$",
    "n": "n",
}
