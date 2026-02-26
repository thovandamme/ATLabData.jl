module DataStructures

export AbstractData
export Grid, ScalarData, VectorData, AveragesData, PlaneData
export FieldHeader, PlanesHeader


""" Parent abstract type for all composite types containing the data """
abstract type AbstractData{T<:AbstractFloat, I<:Signed} end
# abstract type AbstractHeader{T<:AbstractFloat, I<:Signed} end


mutable struct Grid{T<:AbstractFloat,I<:Signed}
    nx::I
    ny::I
    nz::I
    scalex::T
    scaley::T
    scalez::T
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
end
Grid(;
    nx::Signed, ny::Signed, nz::Signed, 
    lx::AbstractFloat, ly::AbstractFloat, lz::AbstractFloat,
    x::Vector{<:AbstractFloat}, 
    y::Vector{<:AbstractFloat}, 
    z::Vector{<:AbstractFloat}
) = Grid(nx, ny, nz, lx, ly, lz, x, y, z)


@kwdef mutable struct FieldHeader{T<:AbstractFloat,I<:Signed}
    headersize::I
    nx::I
    ny::I
    nz::I
    iteration::I
    time::T
    params::Vector{T}
end


@kwdef mutable struct PlanesHeader{T<:AbstractFloat,I<:Signed}
    headersize::I
    iteration::I
    time::T
    planes::Vector{I}
end


@kwdef mutable struct ScalarData{T,I} <: AbstractData{T,I}
    name::String
    grid::Grid{T,I}
    iteration::Int32 = Int32(-1)
    time::T
    field::Array{T,3}
end


# @kwdef mutable struct ScalarData{T,I} <: AbstractData{T,I}
#     name::String
#     header::FieldHeader{T,I}
#     grid::Grid{T,I}
#     field::Array{T,3}
# end


@kwdef mutable struct VectorData{T,I} <: AbstractData{T,I}
    name::String
    grid::Grid{T,I}
    iteration::Int32 = Int32(-1)
    time::T
    field::Array{T,4}
end


# @kwdef mutable struct VectorData{T,I} <: AbstractData{T,I}
#     name::String
#     header::FieldHeader{T,I}
#     grid::Grid{T,I}
#     field::Array{T,4}
# end


mutable struct AveragesData{T,I} <: AbstractData{T,I}
    name::String
    time::Vector{T}
    grid::Grid{T,I}
    field::Array{T,2}
end
AveragesData(;
    name::String,
    time::Vector{<:AbstractFloat},
    z::Vector{<:AbstractFloat},
    field::Array{<:AbstractFloat, 2}
) = AveragesData(
    name, time, 
    Grid{eltype(time), Int}(
        1, 1, length(z), 
        0.0, 0.0, z[end],
        [0.0], [0.0], z
    ),
    field
)


@kwdef mutable struct PlaneData{T,I} <: AbstractData{T,I}
    name::String
    header::PlanesHeader{T,I}
    grid::Grid{T,I}
    field::Array{T,2}
end


function Base.getindex(data::ScalarData, i::Int, j::Int, k::Int)
    return data.field[i,j,k]
end


end