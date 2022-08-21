class StructureType:
    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == to_type(other).name

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
    for sub in abbr.replace("D-", "DC-").strip("-").split("DC"):
        for c in sub:
            rv.append(to_type(c))
        rv.append(DC)
    rv.pop()  # One extra DC
    return rv
