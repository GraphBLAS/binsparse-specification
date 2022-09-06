import itertools

import pytest

from spz.sparsetype import DC, C, S, abbreviate, from_taco, to_taco, unabbreviate


def test_abbreviate():
    expected = "S-C-S-DC-C-S"
    assert abbreviate([S, C, S, DC, C, S]) == expected
    assert abbreviate(S, C, S, DC, C, S) == expected
    assert abbreviate("S", "C", "S", "DC", "C", "S") == expected
    assert abbreviate(["S", "c", "Sparse", "D", "compressed", "sparse"]) == expected


def test_unabbreviate():
    expected = [S, C, S, DC, C, S]
    assert unabbreviate("S-C-S-DC-C-S") == expected
    assert unabbreviate("SCSDCCS") == expected
    assert unabbreviate("SC-S-D-C-S") == expected


@pytest.mark.parametrize("N", range(1, 9))
def test_from_taco(N):
    compressed = "compressed"
    dense = "dense"
    nonunique = "compressed-nonunique"
    singleton = "singleton"
    options = [compressed, dense, nonunique, singleton]
    results = {}
    for taco in itertools.product(*([options] * N)):
        try:
            structure = tuple(from_taco(taco))
        except ValueError:
            continue
        if structure in results:  # pragma: no cover
            print(structure)
            print(" ", results[structure])
            print(" ", taco)
            raise AssertionError(
                "Multiple TACO structures give the same structure: "
                f"{taco} and {results[structure]} -> {structure}"
            )
        results[structure] = taco
        assert tuple(to_taco(structure)) == taco
    assert len(results) == 3 ** (N - 1)
