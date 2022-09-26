from itertools import zip_longest


class StructureType:
    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return other is not None and self.name == to_type(other).name

    def __hash__(self):
        return hash(self.name)

    def __reduce__(self):
        return self.name


# Singletons
class sparse(StructureType):
    name = "sparse"
    abbreviation = "S"


class compressed(StructureType):
    name = "compressed"
    abbreviation = "C"


class doubly_compressed(StructureType):
    name = "doubly_compressed"
    abbreviation = "DC"


S = sparse = sparse()
C = compressed = compressed()
DC = doubly_compressed = doubly_compressed()

_STR_TO_TYPE = {
    "s": S,
    "sparse": S,
    "singleton": S,
    "c": C,
    "compressed": C,
    "dc": DC,
    "d": DC,
    "doubly_compressed": DC,
    "doubly compressed": DC,
    "doubly-compressed": DC,
    "doublycompressed": DC,
}


def to_type(x):
    """Convert a string to a StructureType"""
    if isinstance(x, StructureType):
        return x
    return _STR_TO_TYPE[x.lower()]


def to_str(x):
    return to_type(x).name


def abbreviate(*types):
    if len(types) == 1 and not isinstance(types[0], (StructureType, str)):
        types = types[0]
    abbvs = [to_type(x).abbreviation for x in types]
    sep = "-" if "DC" in abbvs else ""
    return sep.join(abbvs)


def unabbreviate(abbr):
    rv = []
    for sub in abbr.replace("D-", "DC-").replace("-", "").split("DC"):
        for c in sub:
            rv.append(to_type(c))
        rv.append(DC)
    rv.pop()  # One extra DC
    return rv


def to_taco(structure):
    if isinstance(structure, str):
        structure = unabbreviate(structure)
    else:
        structure = unabbreviate(abbreviate(*structure))
    rv = []
    L = [DC] + structure
    lookahead = S  # backwards-fill S values
    for prev, cur, nxt in reversed(list(zip_longest(L[:-1], L[1:], L[2:]))):
        if cur == C:
            rv.append("dense")
        elif prev == S and cur in {S, DC}:
            rv.append("singleton")
        elif prev in {C, DC} and cur == S and nxt is not None and lookahead in {C, S}:
            rv.append("compressed-nonunique")
        elif prev in {C, DC}:
            rv.append("compressed")
        else:
            # We should be able to always go to TACO
            raise NotImplementedError(f"Unable to convert to TACO structure: {structure}")
        if cur != S:
            lookahead = cur
    rv.reverse()

    # Make some assertions about how we interpret TACO structure
    assert rv[0] != "singleton"
    assert rv[-1] != "compressed-nonunique"
    # "compressed-nonunique" may be followed by any number of "singleton" dimensions
    # and then "dense" or end of structure.  E.g., "CN-D", "CN-S-D", and "CN-S-S".
    is_nonunique = False
    for item in rv:
        if is_nonunique and item not in {"singleton", "dense"}:  # pragma: no cover
            raise RuntimeError("Bad TACO format")
        if item == "compressed-nonunique":
            is_nonunique = True
        elif item != "singleton":
            is_nonunique = False

    return rv


def from_taco(structure):
    # We choose a 1-to-1 mapping to and from TACO formats.
    # It's possible (even likely) that multiple TACO formats could technically
    # map to the same format, but we choose 1-to-1 for clarity.
    compressed = "compressed"
    dense = "dense"
    nonunique = "compressed-nonunique"
    singleton = "singleton"
    rv = []
    prev_nonS = structure[0]
    # fmt: off
    for prev, cur, nxt in zip_longest([None] + list(structure[:-1]), structure, structure[1:]):
        # These rules were developed via trial and error.  Fingers crossed!
        # Let's try come up with a clearer way to convert from taco.
        if cur == dense and nxt in {dense, nonunique, compressed}:
            rv.append(C)
        elif (
            prev in {None, dense, singleton, compressed} and cur == compressed
            and nxt in {dense, nonunique, compressed}
            or cur == singleton and (
                prev == compressed and nxt in {dense, nonunique, compressed}
                or prev == singleton and nxt == compressed
                or prev == singleton and nxt in {dense, nonunique} and prev_nonS == compressed
            ) and not (prev == singleton and nxt == compressed and prev_nonS == nonunique)
        ):
            rv.append(DC)
        elif (
            cur == nonunique and nxt in {dense, singleton}
            or cur == compressed and nxt in {None, singleton}
            or cur == singleton and (
                prev == nonunique and nxt in {None, dense, singleton}
                or prev == compressed and nxt == singleton
                or prev == singleton and nxt in {None, singleton}
                or prev == singleton and nxt == dense and prev_nonS == nonunique
            ) and not (nxt is None and prev_nonS == compressed)
        ):
            rv.append(S)
        else:
            raise ValueError(f"Unable to convert from TACO structure: {structure}")
        if cur != "singleton":
            prev_nonS = cur
    # fmt: on
    return rv
