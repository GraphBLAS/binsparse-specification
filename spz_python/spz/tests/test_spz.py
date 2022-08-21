import itertools

import pandas as pd
import pytest

from spz import SPZ


@pytest.fixture
def indices1():
    return [
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 2, 0, 2, 0, 0, 1, 2],
    ]


@pytest.mark.parametrize("shape", [[2, 2, 2, 3], [5, 6, 7, 8], [8, 7, 6, 5]])
def test_4d(indices1, shape):
    df = pd.DataFrame(indices1).T.sort_values([0, 1, 2, 3])
    sparsities = ["S", "C", "DC"]
    for sparsity in itertools.product(sparsities, sparsities, sparsities):
        structure = "".join(sparsity) + "S"
        spz = SPZ(indices1, shape, structure)
        # spz._validate()
        df2 = pd.DataFrame(spz.arrays).T
        pd.testing.assert_frame_equal(df, df2)
