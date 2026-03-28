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
export gradient, gradient!, curl, curl!, divergence, divergence!


let

do_verbose(type::String) = begin
    println("Calculating $type with $(Threads.nthreads()) threads.")
end

################################################################################
#                   Simple derivatives in one direction
################################################################################
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
#                               Curl
################################################################################
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
    res = Array{T}(undef, 3, data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        divergence3D!(res, data, stencil_size)
    else
        divergence2D!(res, data)
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
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose && do_verbose("divergence")
    if data.grid.ny > 1
        divergence3D!(res.field, data, stencil_size)
    else
        divergence2D!(res.field, data)
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
    buffer = similar(res)
    fornberg_method_x!(res, data.field, weights_x, stencil_size)
    fornberg_method_y!(buffer, data.field, weights_y, stencil_size)
    res .+= buffer
    fornberg_method_z!(buffer, data.field, weights_z, stencil_size)
    res .+= buffer
    return nothing
end


function divergence2D!(
        res::Array{T}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    weights_x = get_weights(data.grid.x, stencil_size)
    weights_z = get_weights(data.grid.z, stencil_size)
    buffer = similar(res)
    fornberg_method_x!(res, data.field, weights_x, stencil_size)
    fornberg_method_z!(buffer, data.field, weights_z, stencil_size)
    res .+= buffer
    return nothing
end

end # end of let scope    


end