"""Figures and tables for the Scientific Data descriptor.

``fig01``-``fig06`` are the descriptor's main figures, in its own numbering;
``pixel_series`` and ``vg_vs_direct`` are supporting analyses that render on
request. All of them draw through ``style``, which carries Nature's artwork
specification -- widths in millimetres, the 7 pt type ceiling, the shared
palette and the ``save`` that writes at the declared size. ``run`` is the single
driver (stage 08) and ``basemap`` holds the cartography the map figures share.

``distributions`` and ``summaries`` back Table 1 rather than a figure. They are
run standalone from their own ``__main__`` blocks, have no caller in the package
and are not on ``style`` -- the refactor plan flags them for a second look.
"""

# Shared axis labels. Lived in retention_curve/__init__.py before the package
# was assembled; only figure code has ever used them.
PARAM_SYMBOLS = {
    "theta_r": r"$\theta_r$",
    "theta_s": r"$\theta_s$",
    "alpha": r"$\alpha$",
    "n": "n",
}
