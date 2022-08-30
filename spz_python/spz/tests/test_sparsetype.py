from spz.sparsetype import DC, C, S, abbreviate, unabbreviate


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
