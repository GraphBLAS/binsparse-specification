# Binary Sparse Format Specification
This is part of a new effort to create a binary storage format for storing sparse matrices and other sparse data to disk.

Minutes from our meetings are available [here](https://hackmd.io/0qzK4fJlQp-78t067yiYsA?view) (see also: [previous minutes](minutes)).

## Specification

The working version of the specification can be found under `spec/latest/index.bs`.

## Parsers

Here is a table listing the current tensor frameworks that support the format:

| Language | Framework | Status | Notes |
| -------- | ------ | ------ | ----- |
| C++ | [binsparse-reference-impl](https://github.com/GraphBLAS/binsparse-reference-impl) | under development | converts between binsparse V1.0 and custom in-memory sparse matrices | 
| Julia | [Finch.jl](https://willowahrens.io/Finch.jl/dev/fileio/) | under development | converts between binsparse V1.0 and V2.0 and Finch matrices and tensors |
| Python | [binsparse-python](https://github.com/ivirshup/binsparse-python) | under development | converts between binsparse V1.0 and scipy.sparse matrices |

The working version of the specification can be found under `spec/latest/index.bs`.

### Editing

The spec is written in [bikeshed](https://github.com/tabatkins/bikeshed) – a variant of markdown.
To render the spec locally:

* Install bikeshed (ideally in an isolated environment): `pipx install bikeshed`
* Call `bikeshed spec spec/latest/index.bs`

Rendered versions will generated for pull requests.
