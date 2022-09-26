import itertools

import pytest

from sparsetensorviz.sparsetype import DC, C, S, abbreviate, from_taco, to_taco, unabbreviate


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
        assert tuple(to_taco(structure)) == taco, (taco, structure)
    assert len(results) == 3 ** (N - 1)


# It's not strictly necessary to have both `test_from_taco` and `test_to_taco`,
# but if changes are made, then it's helpful to have both to help debug.
@pytest.mark.parametrize("N", range(1, 9))
def test_to_taco(N):
    options = [DC, C, S]
    results = {}
    for structure in itertools.product(*([options] * (N - 1)), [S]):
        taco = tuple(to_taco(structure))
        if taco in results:  # pragma: no cover
            print(taco)
            print(" ", results[taco])
            print(" ", structure)
            raise AssertionError(
                "Multiple TACO structures give the same structure: "
                f"{taco} and {results[structure]} -> {structure}"
            )
        results[taco] = structure
        assert tuple(from_taco(taco)) == structure, (structure, taco)
    assert len(results) == 3 ** (N - 1)
