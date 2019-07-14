import os

__all__ = ["sep", 'teff_sun', 'log_g_sun', 'dnu_sun', 'numax_sun']

sep = "\\" if os.name=="nt" else "/"

teff_sun = 5777 #k
log_g_sun = 4.437 #dex
dnu_sun = 134.9 #microHz
numax_sun = 3050 # microHz
