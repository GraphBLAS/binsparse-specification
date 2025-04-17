import re

import toolz

from ._core import SparseTensor, unabbreviate


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


class AllSparse(MatcherBase):
    """[S-]S"""

    pattern = re.compile("^S(?P<S>(-S)*)$")

    def __new__(cls, ma, s, *, abbreviate=False):
        numS = num(ma, "S") + 1
        if abbreviate:
            return [("S", numS)]
        return [f"Sparse({numS})"]


class SparseFull(MatcherBase):
    """[S-][F-]F"""

    pattern = re.compile("^(?P<S>(S-)*)(?P<F>(F-)*)F$")

    def __new__(cls, ma, s, *, abbreviate=False):
        numS = num(ma, "S")
        numF = num(ma, "F") + 1
        if abbreviate:
            return [("S", numS), ("D", numF)]
        return [f"Sparse({numS})", f"Dense({numF})"]


class Sparse(MatcherBase):
    """[S-]DC-"""

    pattern = re.compile("^(?P<S>(S-)*)DC-")

    def __new__(cls, ma, s, *, abbreviate=False):
        numS = num(ma, "S") + 1
        if abbreviate:
            return [("S", numS)]
        return [f"Sparse({numS})"]


class Dense(MatcherBase):
    """[C-]C-"""

    pattern = re.compile("^(?P<C>(C-)+)")

    def __new__(cls, ma, s, *, abbreviate=False):
        numC = num(ma, "C")
        if abbreviate:
            return [("D", numC)]
        return [f"Dense({numC})"]


class SparseCompressed(MatcherBase):
    """[S-]S-[C-]C-"""

    pattern = re.compile("^(?P<S>(S-)+)(?P<C>(C-)+)")

    def __new__(cls, ma, s, *, abbreviate=False):
        numS = num(ma, "S")
        numC = num(ma, "C")
        if abbreviate:
            return [("SE", numS), ("D", numC)]
        return [f"Sparse({numS}, expanded=1)", f"Dense({numC})"]


def to_binsparse_groups(s, abbreviate=False):
    if isinstance(s, SparseTensor):
        s = s.abbreviation
    elif not isinstance(s, str):
        raise TypeError(
            f"s argument to to_bundled_groups should be str or SparseTensor; got {type(s)}"
        )
    if "-" not in s:
        s = "-".join(s)
    orig_s = s
    rv = []
    matchers = [
        Sparse,  # [S-]DC-                  -> Sparse(N)
        Dense,  # [C-]C-                    -> Dense(N)
        SparseCompressed,  # [S-]S-[C-]C-   -> Sparse(M, expanded=1), Dense(N)
        # Terminal patterns
        AllSparse,  # [S-]S$                -> Sparse(N)
        SparseFull,  # [S-][F-]F$           -> Sparse(M), Dense(N)
    ]
    while s:
        for matcher in matchers:
            if ma := matcher.match(s):
                rv.extend(matcher(ma, s, abbreviate=abbreviate))
                s = trim(ma, s)
                break
        else:  # pragma: no cover
            raise ValueError(f"Invalid structure {orig_s!r}; unable to handle {s!r}")
    return rv


def structure_from_binsparse(binsparse_structure):
    """Convert user-input binsparse stucture to an internal structure"""
    # This is super quick and sloppy! It allows some very sloppy input
    if not isinstance(binsparse_structure, str):
        text = "-".join(
            val if isinstance(val, str) else "".join(map(str, val)) for val in binsparse_structure
        )
    else:
        text = binsparse_structure
    # Step 1: tokenize input string
    token_map = {
        "sparseexpanded": "SE",
        "expandedsparse": "SE",
        "sparse": "S",
        "se": "SE",
        "es": "SE",
        "s": "S",
        "dense": "D",
        "d": "D",
        "expanded": "E",  # to handle `Sparse(3, expanded=1)`
    }
    ignore = "- []()_=,"
    tokens = []
    t = text.lower()
    while t:
        cont = False
        if t[0] in ignore:
            t = t[1:]
            continue
        for k, v in token_map.items():
            if t.startswith(k):
                tokens.append(v)
                t = t[len(k) :]
                cont = True
                break
        if cont:
            continue
        for i, c in enumerate(t):
            if not c.isdecimal():
                if i == 0:
                    raise ValueError(f"Bad input: {binsparse_structure}")
                tokens.append(int(t[:i]))
                t = t[i:]
                cont = True
                break
        if cont:
            continue
        if t.isdecimal():
            tokens.append(int(t))
            break
        raise ValueError(f"Bad input: {binsparse_structure}")

    # Step 2: process tokens to form canonical binsparse format (abbreviated)
    levels = []
    it = toolz.sliding_window(4, tokens + [None] * 3)
    for cur, n1, n2, n3 in it:
        if cur == "D":
            if isinstance(n1, int):
                next(it)
            else:
                n1 = 1
            levels.append(("D", n1))
        elif cur == "S":
            if isinstance(n1, int):
                next(it)
            else:
                n1, n2, n3 = 1, n1, n2
            if n2 != "E":
                levels.append(("S", n1))
            else:
                next(it)
                if isinstance(n3, int):
                    if n3 != 1:
                        raise ValueError(f"Bad input: {binsparse_structure}")
                    next(it)
                levels.append(("SE", n3))
        elif cur == "SE":
            if isinstance(n1, int):
                next(it)
                if n1 != 1:
                    raise ValueError(f"Bad input: {binsparse_structure}")
            else:
                n1 = 1
            levels.append(("SE", n1))
        else:
            raise ValueError(f"Bad input: {binsparse_structure}")

    for i, (cur, n) in enumerate(levels):
        if cur == "SE":
            if len(levels) == i + 1 or levels[i + 1][0] != "D":
                raise ValueError(f"Sparse({n}, expanded=1) level must be followed by a Dense level")
            if n != 1:
                raise ValueError(f"Bad input: {binsparse_structure}")
    if levels[-1][0] == "D":
        raise ValueError("Sparse structure must end in Sparse level; got Dense")

    # Step 3: convert to internal levels
    converted_levels = []
    for level, n in levels:
        if level == "D":
            converted_levels.extend(["C"] * n)
        elif level == "S":
            converted_levels.extend(["S"] * (n - 1))
            converted_levels.append("DC")
        else:  # level == "SE"
            converted_levels.extend(["S"] * n)
    assert converted_levels[-1] == "DC"
    converted_levels[-1] = "S"
    return unabbreviate("-".join(converted_levels))
