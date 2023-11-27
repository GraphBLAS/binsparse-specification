#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Finch
using MatrixMarket
using MatrixDepot
using TensorMarket
using HDF5
using JSON
using SparseArrays

for (vec_key, x) in [
    "foo" => [11, 12, 13, 14],
    "bar" => sparse([0, 0, 13, 0, 0, 16, 0, 0, 19]),
]
    for (fmt_key, fmt) in [
        "VEC" => Fiber!(SparseList(Element(zero(eltype(x))))),
        "DVEC" => Fiber!(Dense(Element(zero(eltype(x))))),
    ]
        fmt = dropdefaults!(fmt, x)
        example_dir = joinpath(@__DIR__, "reference", "$(vec_key)_$(fmt_key)")
        mkpath(example_dir)
        fwrite(joinpath(example_dir, "$(vec_key).mtx"), x)
        fwrite(joinpath(example_dir, "$(vec_key)_$(fmt_key).bsp.h5"), fmt)
    end
end

for (mtx_key, A) in [
    "mycielskian3" => SparseMatrixCSC(matrixdepot("Mycielski/mycielskian3")),
    "b1_ss" => SparseMatrixCSC(matrixdepot("Grund/b1_ss")),
    "farm" => SparseMatrixCSC(matrixdepot("Meszaros/farm")),
]
    for (fmt_key, fmt) in [
        "CSR" => swizzle(Fiber!(Dense(SparseList(Element(zero(eltype(A)))))), 2, 1),
        "CSC" => Fiber!(Dense(SparseList(Element(zero(eltype(A)))))),
        "DMAT" => swizzle(Fiber!(Dense(Dense(Element(zero(eltype(A)))))), 2, 1),
        "DMATR" => Fiber!(Dense(Dense(Element(zero(eltype(A)))))),
        "DCSR" => swizzle(Fiber!(SparseList(SparseList(Element(zero(eltype(A)))))), 2, 1),
        "DCSC" => Fiber!(SparseList(SparseList(Element(zero(eltype(A)))))),
        "COO" => swizzle(Fiber!(SparseCOO{2}(Element(zero(eltype(A))))), 2, 1),
        "COOC" => Fiber!(SparseCOO{2}(Element(zero(eltype(A))))),
    ]
        fmt = copyto!(fmt, A)
        example_dir = joinpath(@__DIR__, "reference", "$(mtx_key)_$(fmt_key)")
        mkpath(example_dir)
        fwrite(joinpath(example_dir, "$(mtx_key).mtx"), A)
        fwrite(joinpath(example_dir, "$(mtx_key)_$(fmt_key).bsp.h5"), fmt)
    end
end