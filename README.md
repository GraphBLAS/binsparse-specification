# Binary Sparse Format Specification
This is part of a new effort to create a binary storage format for storing sparse matrices and other sparse data to disk.

Minutes from our meetings are available [here](https://hackmd.io/0qzK4fJlQp-78t067yiYsA?view) (see also: [previous minutes](minutes)).



## Specification

[View Latest Spec](https://graphblas.org/binsparse-specification/)

## Parsers

Here is a table listing the current tensor frameworks that support the format:

| Language | Framework | Status | Notes |
| -------- | ------ | ------ | ----- |
| C | [binsparse-reference-c](https://github.com/GraphBLAS/binsparse-reference-c) | under development | converts between binsparse and in-memory sparse matrices |
| C++ | [binsparse-reference-impl](https://github.com/GraphBLAS/binsparse-reference-impl) | under development | converts between binsparse and in-memory sparse matrices | 
| Julia | [Finch.jl](https://willowahrens.io/Finch.jl/dev/fileio/) | under development | converts between binsparse and Finch matrices and tensors, supports custom format spec |
| Python | [binsparse-python](https://github.com/ivirshup/binsparse-python) | under development | converts between binsparse and scipy.sparse matrices |

### Editing

The working version of the specification can be found under `spec/latest/index.bs`.

The spec is written in [bikeshed](https://github.com/tabatkins/bikeshed) – a variant of markdown.
To render the spec locally:

* Install bikeshed (ideally in an isolated environment): `pipx install bikeshed`
* Call `bikeshed spec spec/latest/index.bs`

To render the spec online, see [api.csswg.org](https://api.csswg.org/bikeshed/),
or use the following commands:

```
cd binsparse-specification/spec/latest
curl https://api.csswg.org/bikeshed/ -F file=@index.bs > index.html
```

Rendered versions will generated for pull requests.
