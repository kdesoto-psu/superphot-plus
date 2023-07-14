import os

import numpy as np

from superphot_plus.ztf_transient_fit import dynesty_single_file


def test_dynesty_single_file(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with fits"""
    dynesty_single_file(
        single_ztf_lightcurve_compressed, tmp_path, skip_if_exists=False,
        rstate=np.random.default_rng(9876)
    )

    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_dynesty.npz")
    assert os.path.exists(output_file)

    params = np.load(output_file)["arr_0"]
    ## could be between ~600 and ~800, but hits 667 with chosen seed.
    assert len(params) == 667