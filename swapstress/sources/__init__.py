"""Harmonization of every input source to (theta, suction_cm, depth_cm).

One module per source. ``<source>.py`` reads and standardizes the observations;
``<source>_sites.py`` builds the station geometry and its MGRS tile join, which
is what Earth Engine extraction samples and what the spatial split blocks on.
"""
