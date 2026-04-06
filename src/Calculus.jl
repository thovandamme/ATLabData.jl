module Calculus

# This module does calculus operation on ScalarData and VectorData and 
# returns the same. It is intended as convenient option in the ATLabData.jl
# framework. Differentiation operation on simple arrays are located at FDM.jl.

# The stencil size defaults to 7 everywhere and should only be optional in the 
# api functions inside this module.

# Data comes usually from the files as Float32 and so far numerical operations
# work with that precission.
# Perhaps do the calculus in Float64, if memory is available?

using ..DataStructures
using ..FDM
using Polyester

export ∂x, ∂y, ∂z, ∂x!, ∂y!, ∂z!
export ∂x², ∂y², ∂z², ∂x2, ∂y2, ∂z2
export gradient, gradient!, jacobian, jacobian!
export curl, curl!, divergence, divergence!


let

do_verbose(type::String) = begin
    println("Calculating $type with $(Threads.nthreads()) threads.")
end

################################################################################
#                   Simple derivatives in one direction
################################################################################

# Derivatives of 1D data
global function ∂x(
        data::Vector{T}, axis::Vector{T}; stencil_size=7, verbose=true
    )::Vector{T} where {T<:AbstractFloat}
    verbose && do_verbose("1D derivative")
    res = similar(data)
    weights = get_weights(axis, stencil_size)
    stencils = get_stencils(length(axis), stencil_size)
    fornberg_method_1D!(res, data, weights, stencils)
    return res
end


global function ∂x²(
        data::Vector{T}, axis::Vector{T}; stencil_size=7, verbose=true
    )::Vector{T} where {T<:AbstractFloat}
    verbose && do_verbose("2nd-order 1D derivative")
    res = similar(data)
    weights = get_weights(axis, stencil_size, order=2)
    stencils = get_stencils(length(axis), stencil_size)
    fornberg_method_1D!(res, data, weights, stencils)
    return res
end
global ∂x2(args...; kwargs...) = ∂x²(args...; kwargs...)


global ∂y(
    data::Vector{T}, axis::Vector{T}; stencil_size=7, verbose=true
) where {T<:AbstractFloat} = ∂x(data, axis, stencil_size=stencil_size, verbose=verbose)
global ∂y²(
    data::Vector{T}, axis::Vector{T}; stencil_size=7, verbose=true
) where {T<:AbstractFloat} = ∂x²(data, axis, stencil_size=stencil_size, verbose=verbose)
global ∂y2(args...; kwargs...) = ∂y²(args...; kwargs...)


global ∂z(
    data::Vector{T}, axis::Vector{T}; stencil_size=7, verbose=true
) where {T<:AbstractFloat} = ∂x(data, axis, stencil_size=stencil_size, verbose=verbose)
global ∂z²(
    data::Vector{T}, axis::Vector{T}; stencil_size=7, verbose=true
) where {T<:AbstractFloat} = ∂x²(data, axis, stencil_size=stencil_size, verbose=verbose)
global ∂z2(args...; kwargs...) = ∂z²(args...; kwargs...)


# Derivatives of ScalarData
global function ∂x(
        data::ScalarData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("derivative in x")
    res = similar(data.field)
    weights = get_weights(data.grid.x, stencil_size)
    stencils = get_stencils(data.grid.nx, stencil_size)
    fornberg_method_x!(res, data.field, weights, stencils)
    return ScalarData(
        name = "∂₁($(data.name))",
        grid = data.grid,
        iteration = data.iteration,
        time = data.time,
        field = res
    )
end


global function ∂x!(
        res::ScalarData{T,I}, data::ScalarData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("derivative in x")
    weights = get_weights(data.grid.x, stencil_size)
    stencils = get_stencils(data.grid.nx, stencil_size)
    fornberg_method_x!(res.field, data.field, weights, stencils)
    return nothing
end


global function ∂y(
        data::ScalarData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("derivative in y")
    res = similar(data.field)
    weights = get_weights(data.grid.y, stencil_size)
    stencils = get_stencils(data.grid.ny, stencil_size)
    fornberg_method_y!(res, data.field, weights, stencils)
    return ScalarData(
        name = "∂₂($(data.name))",
        grid = data.grid,
        iteration = data.iteration,
        time = data.time,
        field = res
    )
end


global function ∂y!(
        res::ScalarData{T,I}, data::ScalarData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("derivative in y")
    weights = get_weights(data.grid.y, stencil_size)
    stencils = get_stencils(data.grid.ny, stencil_size)
    fornberg_method_y!(res.field, data.field, weights, stencils)
    return nothing
end


global function ∂z(
        data::ScalarData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("derivative in z")
    res = similar(data.field)
    weights = get_weights(data.grid.z, stencil_size)
    stencils = get_stencils(data.grid.nz, stencil_size)
    fornberg_method_z!(res, data.field, weights, stencils)
    return ScalarData(
        name = "∂₃($(data.name))",
        grid = data.grid,
        iteration = data.iteration,
        time = data.time,
        field = res
    )
end


global function ∂z!(
        res::ScalarData{T,I}, data::ScalarData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("derivative in z")
    weights = get_weights(data.grid.z, stencil_size)
    stencils = get_stencils(data.grid.nz, stencil_size)
    fornberg_method_z!(res.field, data.field, weights, stencils)
    return nothing
end


################################################################################
#                           Gradient
################################################################################
"""
    gradient(data)
"""
global function gradient(
        data::ScalarData{T,I}; stencil_size::Signed=7, verbose=true
    )::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("gradient")
    res = Array{T}(undef, length(size(data.field)), data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        gradient3D!(res, data, stencil_size)
    else
        gradient2D!(res, data, stencil_size)
    end    
    return VectorData(
        name = "∇($(data.name))",
        grid = data.grid,
        time = data.time,
        field = res
    )
end


global function gradient!(
        res::VectorData{T,I},
        data::ScalarData{T,I};
        stencil_size::Signed = 7,
        verbose = true
    ) where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("gradient")
    if data.grid.ny > 1
        gradient3D!(res.field, data, stencil_size)
    else
        gradient2D!(res.field, data, stencil_size)
    end
    res.name = "∇($(data.name))"
    return nothing
end


function gradient3D!(
        res::Array{T,4}, data::ScalarData{T,I}, stencil_size::Signed
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_y = get_weights(data.grid.y, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    x_stencils = get_stencils(data.grid.nx, stencil_size)
    y_stencils = get_stencils(data.grid.ny, stencil_size)
    z_stencils = get_stencils(data.grid.nz, stencil_size)
    fornberg_method_x!(view(res, 1, :, :, :), data.field, weights_x, x_stencils)
    fornberg_method_y!(view(res, 2, :, :, :), data.field, weights_y, y_stencils)
    fornberg_method_z!(view(res, 3, :, :, :), data.field, weights_z, z_stencils)
    return nothing
end


function gradient2D!(
        res::Array{T,4}, data::ScalarData{T,I}, stencil_size::Signed
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    x_stencils = get_stencils(data.grid.nx, stencil_size)
    z_stencils = get_stencils(data.grid.nz, stencil_size)
    fornberg_method_x!(view(res, 1, :, :, :), data.field, weights_x, x_stencils)
    fornberg_method_z!(view(res, 2, :, :, :), data.field, weights_z, z_stencils)
    return nothing
end


################################################################################
#                   Jacobian (gradient of vector field)
################################################################################
"""
    jacobian!(res, field, grid)
Mutating variant of jacobian(). Serves as kernel for all other methods of 
jacobian().
"""
# TODO proper implementation for 2D
global function jacobian!(
        res::AbstractArray{T,5}, field::AbstractArray{T,4}, grid::Grid{T,I}; 
        stencil_size::Signed=9, verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    twoD = false
    (size(field)[3]==1) && (twoD=true)
    verbose && do_verbose("jacobian")
    weights_x = get_weights(grid.x, stencil_size)
    (! twoD) && (weights_y = get_weights(grid.y, stencil_size))
    # weights_y = get_weights(grid.y, stencil_size)
    weights_z = get_weights(grid.z, stencil_size)
    x_stencils = get_stencils(grid.nx, stencil_size)
    y_stencils = get_stencils(grid.ny, stencil_size)
    z_stencils = get_stencils(grid.nz, stencil_size)
    # Loop over the vector field components
    for h ∈ 1:3
        fornberg_method_x!(
            view(res, 1, h, :, :, :), view(field, h, :, :, :),
            weights_x, x_stencils
        )
        if twoD
            view(res, 2, h, :, :, :) .= 0.0
        else
            fornberg_method_y!(
                view(res, 2, h, :, :, :), view(field, h, :, :, :),
                weights_y, y_stencils
            )
        end
        fornberg_method_z!(
            view(res, 3, h, :, :, :), view(field, h, :, :, :),
            weights_z, z_stencils
        )
    end
    return nothing
end

# This functions trys to reduce the multithread overhead
# global function jacobian3D!(
#         res::AbstractArray{T,5}, field::AbstractArray{T,4}, grid::Grid{T,I}; 
#         stencil_size::Signed=9, verbose=true
#     ) where {T<:AbstractFloat, I<:Signed}
#     verbose && do_verbose("jacobian")
#     weights_x = get_weights(grid.x, stencil_size)
#     weights_y = get_weights(grid.y, stencil_size)
#     weights_z = get_weights(grid.z, stencil_size)
#     x_stencils = get_stencils(grid.nx, stencil_size)
#     y_stencils = get_stencils(grid.ny, stencil_size)
#     z_stencils = get_stencils(grid.nz, stencil_size)
    
#     # Merged loop for derivatives in x and z
#     @inbounds @batch for k ∈ 1:nz
        
#     end
#     return nothing
# end


"""
    jacobian!(res, data)
Mutating variant of jacobian(data).
"""
global function jacobian!(
        res::AbstractArray{T,5}, data::VectorData{T,I}; stencil_size::Signed=9, verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    jacobian!(res, data.field, data.grid, stencil_size=stencil_size, verbose=verbose)
    return nothing
end


"""
    jacobia(data)
Calculate the Jacobian of the VectorData data. grid of type Grid contains 
the axis information.
"""
global function jacobian(
        data::VectorData{T,I}; stencil_size::Signed=9, verbose=true
    )::Array{T,5} where {T<:AbstractFloat, I<:Signed}
    return jacobian(data.field, data.grid, stencil_size=stencil_size, verbose=verbose)
end


"""
    jacobia(field, grid)
Calculate the Jacobian of the vector valued field. grid of type Grid contains 
the axis information.
"""
global function jacobian(
        field::AbstractArray{T,4}, grid::Grid{T,I}; stencil_size::Signed=9, verbose=true
    )::Array{T,5} where {T<:AbstractFloat, I<:Signed}
    res = Array{T}(undef, 3, 3, grid.nx, grid.ny, grid.nz)
    jacobian!(res, field, grid, stencil_size=stencil_size, verbose=verbose)
    return res
end


################################################################################
#                               Curl
################################################################################
# TODO Implement with FDM module
"""
    curl(data)
Calculates ∇×(data) and returns VectorData.
"""
global function curl(
        data::VectorData{T,I}; stencil_size=7, verbose=true
    )::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("curl")
    res = Array{T}(undef, 3, data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        curl3D!(res, data, stencil_size)
    else
        curl2D!(res, data, stencil_size)
    end    
    return VectorData(
        name = "∇×($(data.name))",
        grid = data.grid,
        time = data.time,
        field = res
    )
end


"""
    curl!(res, data)
Mutating variant fo curl(data).
"""
global function curl!(
        res::VectorData{T,I}, data::VectorData{T,I}; stencil_size=7, verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("curl")
    if data.grid.ny > 1
        curl3D!(res.field, data, stencil_size)
    else
        curl2D!(res.field, data, stencil_size)
    end
    res.name = "∇×($(data.name))"
    return nothing
end


# TODO
function curl3D!(
        res::Array{T,4}, data::VectorData{T,I}, stencil_size::Signed
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_y = get_weights(data.grid.y, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    x_stencils = get_stencils(data.grid.nx, stencil_size)
    y_stencils = get_stencils(data.grid.ny, stencil_size)
    z_stencils = get_stencils(data.grid.nz, stencil_size)
    @inbounds @batch for k ∈ eachindex(data.grid.z)
        for j ∈ eachindex(data.grid.y)
            for i ∈ eachindex(data.grid.x)
                res[1,i,j,k] = sum(weights_y[i] .* data.field[3,i,y_stencils[j],k])
                res[1,i,j,k] -= sum(weights_z[i] .* data.field[2,i,j,z_stencils[k]])

                res[2,i,j,k] = sum(weights_z[i] .* data.field[1,i,j,z_stencils[k]])
                res[2,i,j,k] -= sum(weights_x[i] .* data.field[3,x_stencils[i],j,k])

                res[3,i,j,k] = sum(weights_x[i] .* data.field[2,x_stencils[i],j,k])
                res[3,i,j,k] -= sum(weights_y[i] .* data.field[1,i,y_stencils[j],k])
            end
        end
    end
end


function curl2D!(
        res::Array{T}, data::VectorData{T,I}, stencil_size::Signed
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    x_stencils = get_stencils(data.grid.nx, stencil_size)
    z_stencils = get_stencils(data.grid.nz, stencil_size)
    @inbounds @batch for k ∈ eachindex(data.grid.z)
        for j ∈ eachindex(data.grid.y)
            for i ∈ eachindex(data.grid.x)
                res[2,i,j,k] = sum(weights_z[i] .* data.field[1,i,j,z_stencils[k]])
                res[2,i,j,k] -= sum(weights_x[i] .* data.field[3,x_stencils[i],j,k])
            end
        end
    end
end


################################################################################
#                           Divergence
################################################################################
global function divergence(
        data::VectorData{T,I}; stencil_size=7, verbose=true
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("divergence")
    res = Array{T}(undef, data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        divergence3D!(res, data, stencil_size)
    else
        divergence2D!(res, data, stencil_size)
    end
    return ScalarData(
        name = "∇⋅($(data.name))",
        grid = data.grid,
        iteration = data.iteration,
        time = data.time,
        field = res
    )
end


global function divergence!(
        res::ScalarData{T,I}, data::VectorData{T,I}; stencil_size=7, verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("divergence")
    if data.grid.ny > 1
        divergence3D!(res.field, data, stencil_size)
    else
        divergence2D!(res.field, data, stencil_size)
    end
    res.name = "∇⋅($(data.name))"
    return nothing
end


function divergence3D!(
        res::Array{T}, data::VectorData{T,I}, stencil_size::Signed
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_y = get_weights(data.grid.y, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    stencils_x = get_stencils(data.grid.nx, stencil_size)
    stencils_y = get_stencils(data.grid.ny, stencil_size)
    stencils_z = get_stencils(data.grid.nz, stencil_size)
    buffer = similar(res)
    fornberg_method_x!(res, view(data.field, 1, :, :, :), weights_x, stencils_x)
    fornberg_method_y!(buffer, view(data.field, 2, :, :, :), weights_y, stencils_y)
    res .+= buffer
    fornberg_method_z!(buffer, view(data.field, 3, :, :, :), weights_z, stencils_z)
    res .+= buffer
    return nothing
end


function divergence2D!(
        res::Array{T}, data::VectorData{T,I}, stencil_size::Signed
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    stencils_x = get_stencils(data.grid.nx, stencil_size)
    stencils_z = get_stencils(data.grid.nz, stencil_size)
    buffer = similar(res)
    fornberg_method_x!(res, view(data.field, 1, :, :, :), weights_x, stencils_x)
    fornberg_method_z!(buffer, view(data.field, 3, :, :, :), weights_z, stencils_z)
    res .+= buffer
    return nothing
end

end # end of let scope    


end