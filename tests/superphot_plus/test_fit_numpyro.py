import os

import numpy as np

from superphot_plus.fit_numpyro import numpyro_single_file


def test_numpyro_nuts(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with samples that can be read."""
    numpyro_single_file(single_ztf_lightcurve_compressed, tmp_path, sampler="NUTS")

    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_NUTS.npz")
    assert os.path.exists(output_file)

    params = np.load(output_file)["arr_0"]
    ## NUTS defaults to 300 samples
    assert len(params) == 300

def test_numpyro_svi(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with samples that can be read."""
    numpyro_single_file(single_ztf_lightcurve_compressed, tmp_path, sampler="svi")

    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_svi.npz")
    assert os.path.exists(output_file)

    params = np.load(output_file)["arr_0"]
    ## svi defaults to 100 samples
    assert len(params) == 100