import numpy as np
from .swrc import SWRC


class ReeshSWRC(SWRC):
    """
    ReESH-style fitting: uses least-squares optimizer and relaxed
    initial parameters with no tight bounds.
    """

    def _generate_initial_params(self, data_for_depth):
        params = self._vg_model.make_params()
        params['theta_r'].set(value=float(np.nanmin(data_for_depth['theta'])))
        params['theta_s'].set(value=float(np.nanmax(data_for_depth['theta'])))
        params['alpha'].set(value=0.01)
        params['n'].set(value=1.2)
        return params

    def fit(self, report=False, method='leastsq'):
        # Force least-squares by default to match ReESH
        return super().fit(report=report, method=method)

