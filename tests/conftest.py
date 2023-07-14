import os
import os.path

import pytest

TEST_DIR = os.path.dirname(__file__)

DATA_DIR_NAME = "data"

# pylint: disable=missing-function-docstring, redefined-outer-name

@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, DATA_DIR_NAME)

@pytest.fixture
def single_ztf_lightcurve(test_data_dir):
    return os.path.join(test_data_dir, "ZTF22abvdwik.csv")


@pytest.fixture
def single_ztf_lightcurve_compressed(test_data_dir):
    return os.path.join(test_data_dir, "ZTF22abvdwik.npz")


@pytest.fixture
def single_ztf_sn_id():
    return "ZTF22abvdwik"