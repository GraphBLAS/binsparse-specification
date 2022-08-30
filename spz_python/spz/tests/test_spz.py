import itertools
import random

import numpy as np
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
def test_indices1(indices1, shape):
    df = pd.DataFrame(indices1).T.sort_values([0, 1, 2, 3])
    sparsities = ["S", "C", "DC"]
    for sparsity in itertools.product(sparsities, sparsities, sparsities):
        structure = "".join(sparsity) + "S"
        spz = SPZ(indices1, shape, structure)
        spz._validate()
        # spz._repr_svg_()
        df2 = pd.DataFrame(spz.arrays).T
        pd.testing.assert_frame_equal(df, df2)


def test_rank1():
    for size in range(1, 4):
        for num in range(1, size + 1):
            for array in itertools.combinations(range(size), num):
                array = np.array(array)
                spz = SPZ([array], (size,), "S")
                spz._validate()
                # spz._repr_svg_()
                [index] = spz.arrays
                assert np.array_equal(array, index)


@pytest.mark.slow
def test_rank2():
    for nrows in range(1, 4):
        for ncols in range(1, 4):
            size = nrows * ncols
            for num in range(1, size + 1):
                for array in itertools.combinations(range(size), num):
                    array = np.array(array)
                    x = array // ncols
                    y = array % ncols
                    for sparsity in ["S", "C", "DC"]:
                        spz = SPZ([x, y], (nrows, ncols), sparsity + "S")
                        spz._validate()
                        # spz._repr_svg_()
                        rows, cols = spz.arrays
                        assert np.array_equal(x, rows)
                        assert np.array_equal(y, cols)


@pytest.mark.slow
def test_rank3():
    for nx in range(1, 3):
        for ny in range(1, 3):
            for nz in range(1, 3):
                size = nx * ny * nz
                for num in range(1, size + 1):
                    for array in itertools.combinations(range(size), num):
                        array = np.array(array)
                        x = array // (ny * nz)
                        y = array // nz % ny
                        z = array % nz
                        for sparsity in itertools.product(["S", "C", "DC"], ["S", "C", "DC"]):
                            if random.random() < 2 / 3:
                                # Randomly skip two thirds of cases to speed up testing
                                continue
                            structure = "".join(sparsity) + "S"
                            spz = SPZ([x, y, z], (nx, ny, nz), structure)
                            spz._validate()
                            # spz._repr_svg_()
                            x2, y2, z2 = spz.arrays
                            assert np.array_equal(x, x2)
                            assert np.array_equal(y, y2)
                            assert np.array_equal(z, z2)


@pytest.mark.slow
@pytest.mark.parametrize("N", range(1, 7))
def test_rankN(N):
    sparsities = [["S", "C", "DC"]] * (N - 1)
    for sparsity in itertools.product(*sparsities):
        structure = "".join(sparsity) + "S"
        # For each structure, randomly choose each dimension to be size 1, 2, 3, or 4
        shape = tuple(random.randint(1, 4) for _ in range(N))
        size = np.multiply.reduce(shape)
        # Randomly choose the number of elements
        num = random.randint(1, size)
        # Randomly add indices for the elements
        flat_idx = set()
        while len(flat_idx) < num:
            flat_idx.add(random.randrange(size))
        array = np.array(sorted(flat_idx))
        if N == 1:
            indices = []
        else:
            indices = [array // int(np.multiply.reduce(shape[1:]))]
        for i in range(1, N - 1):
            indices.append(array // np.multiply.reduce(shape[i + 1 :]) % shape[i])
        indices.append(array % shape[-1])
        try:
            spz = SPZ(indices, shape, structure)
        except Exception:  # pragma: no cover
            print("N:", N)
            print("array:", array)
            print("structure:", structure)
            print("shape:", shape)
            print("indices:", indices)
            raise
        try:
            spz._validate()
        except Exception:  # pragma: no cover
            print("structure:", spz.structure)
            print("shape:", spz.shape)
            print("indices:", spz._indices)
            print("pointers:", spz._pointers)
            print(spz)
            raise
        # spz._repr_svg_()
        arrays = spz.arrays
        for index, arr in zip(indices, arrays):
            assert np.array_equal(index, arr)
