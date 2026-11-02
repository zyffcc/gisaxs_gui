"""gisaxs1d_kit: the GISAXS / SAXS 1D line-cut fitting tool packaged for a GUI.

What a GUI imports (this file, client.py, results.py and qt_panel.py need only the standard library, numpy and,
for the panel, PyQt5 with pyqtgraph or matplotlib):

    from gisaxs1d_kit import Fit1DClient
    client = Fit1DClient()                              # starts one background worker process on first use
    res = client.fit(q, intensity, sigma, q_unit='A^-1', composition='sphere,vc', qz=0.5, policy='fast')
    res['table']        # rows ready for a table widget (parameter, value, error, range, status)
    res['curves']       # q, data, model and components for plotting
    client.close()

    from gisaxs1d_kit.qt_panel import Fit1DPanel         # embeddable QWidget; python -m gisaxs1d_kit.qt_panel runs it alone

The fit itself (network proposals, grid alignment, LM, multi-solution check) runs in the worker process started by
the client, or in a job process of the host application (gisaxs1d_kit.jobs.run_fit1d_job has the signature of a
GIMaP JobRunner handler). Nothing heavy is imported into the GUI process.

Conventions of every number that leaves the kit: q in nm^-1 (input may be given in A^-1), lengths in nm, physics
v2 of the fit1d tool (number-weighted size distribution, Hosemann paracrystal with w = w_rel * D, spheres at
|q| = sqrt(q_par^2 + qz^2)). They are repeated in every result under 'conventions'.
"""
from gisaxs1d_kit._paths import KIT_VERSION, kit_home

__version__ = KIT_VERSION
__all__ = ['Fit1DClient', 'Fit1DError', 'Fit1DCancelled', 'KIT_VERSION', 'kit_home']


def __getattr__(name):                       # the client is imported on first use (keeps `import gisaxs1d_kit` instant)
    if name in ('Fit1DClient', 'Fit1DError', 'Fit1DCancelled'):
        from gisaxs1d_kit import client
        return getattr(client, name)
    raise AttributeError(name)
