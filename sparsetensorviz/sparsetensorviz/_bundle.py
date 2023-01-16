import re

from ._core import SparseTensor


def num(match, name, tokenlength=2):
    s = match.group(name)
    if s is None:
        return 0
    return len(s) // tokenlength


def trim(ma, s):
    start, stop = ma.span(0)
    assert start == 0
    return s[stop:]


class MatcherBase:
    @classmethod
    def match(cls, s):
        return cls.pattern.match(s)


class AllFull(MatcherBase):
    """F[-F]"""

    pattern = re.compile("^F(?P<F>(-F)*)$")

    def __new__(cls, ma, s):
        numF = num(ma, "F")
        return [f"Full({numF + 1})"]


class AllCoord(MatcherBase):
    """S[-S]"""

    pattern = re.compile("^(?P<S>(S-)*)S$")

    def __new__(cls, ma, s):
        numS = num(ma, "S")
        return [f"Coord({numS + 1})"]


class InitSparse(MatcherBase):
    """C-[C-]"""

    pattern = re.compile("^(?P<C>(C-)+)")

    def __new__(cls, ma, s):
        numC = num(ma, "C")
        return [f"Sparse({numC})"]


class CoordSparse(MatcherBase):
    """[S-]DC-C-[C-]"""

    pattern = re.compile("^(?P<S>(S-)*)DC-(?P<C>(C-)+)")

    def __new__(cls, ma, s):
        numS = num(ma, "S")
        numC = num(ma, "C")
        return [f"Coord({numS + 1})", f"Sparse({numC})"]


class CoordSparseExpanded(MatcherBase):
    """[S-]S-C-[C-]"""

    pattern = re.compile("^(?P<S>(S-)+)(?P<C>(C-)+)")

    def __new__(cls, ma, s):
        numS = num(ma, "S")
        numC = num(ma, "C")
        return [f"Coord({numS}, expanded=1)", f"Sparse({numC})"]


class HyperSparse(MatcherBase):
    """[S-]DC-"""

    pattern = re.compile("^(?P<S>(S-)*)DC-")

    def __new__(cls, ma, s):
        numS = num(ma, "S")
        return [f"HyperSparse({numS + 1})"]


class CoordFull(MatcherBase):
    """[S-]S-F[-F]"""

    pattern = re.compile("^(?P<S>(S-)+)F(?P<F>(-F)*)$")

    def __new__(cls, ma, s):
        numS = num(ma, "S")
        numF = num(ma, "F")
        return [f"Coord({numS})", f"Full({numF + 1})"]


def to_bundled_groups(s):
    if isinstance(s, SparseTensor):
        s = s.abbreviation
    elif not isinstance(s, str):
        raise TypeError(
            f"s argument to to_bundled_groups should be str or SparseTnesor; got {type(s)}"
        )
    if "-" not in s:
        s = "-".join(s)
    orig_s = s
    if ma := AllFull.match(s):  # All F
        return AllFull(ma, s)
    if ma := AllCoord.match(s):  # All S
        return AllCoord(ma, s)
    rv = []
    if ma := InitSparse.match(s):  # Begins with C
        rv.extend(InitSparse(ma, s))
        s = trim(ma, s)
    matchers = [
        CoordSparse,  # [S-]DC-C-[C-]
        CoordSparseExpanded,  # [S-]S-C-[C-]
        HyperSparse,  # [S-]DC-
        AllCoord,  # [S-]S
        CoordFull,  # [S-]S-F[-F]
    ]
    while s:
        for matcher in matchers:
            if ma := matcher.match(s):
                rv.extend(matcher(ma, s))
                s = trim(ma, s)
                break
        else:  # pragma: no cover
            raise ValueError(f"Invalid structure {orig_s!r}; unable to handle {s!r}")
    return rv
