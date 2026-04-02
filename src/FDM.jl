module FDM

using Polyester
using LoopVectorization

export get_weights, get_stencils
export fornberg_method_1D!, fornberg_method_x!, fornberg_method_y!, fornberg_method_z!


let
    """
        get_weights(axis, stencil_size)
    Initialize the Fornberg weights along a complete axis with stencil size.
    """
    global function get_weights(
            axis::Vector{T}, stencil_size::Signed; order=1
        )::Vector{Vector{<:AbstractFloat}} where {T<:AbstractFloat}
        nx = length(axis)
        res = Vector{Vector{<:AbstractFloat}}(undef, nx)
        half = stencil_size÷2
        @inbounds for i ∈ eachindex(axis)
            imin = max(1, i-half) # minimum index is 1
            imax = min(nx, i+half) # maximum index is nx
            if imax - imin + 1 < stencil_size
                if i <= half
                    imin, imax = 1, stencil_size
                else
                    imin, imax = nx-stencil_size+1, nx
                end
            end
            res[i] = fdweights(axis[imin:imax] .- axis[i], order)
        end
        return res
    end


    """
        fdweights(t,m)

    Compute weights for the `m`th derivative of a function at zero using
    values at the nodes in vector `t`.
    """
    # --- From "Fundamental of Numerical Computation - Julia Edition" ------
    # https://github.com/fncbook/FundamentalsNumericalComputation.jl
    # https://fncbook.com/finitediffs/#arbitrary-nodes
    function fdweights(t,m)
        # This is a compact implementation, not an efficient one.
        # Recursion for one weight. 
        function weight(t,m,r,k)
            # Inputs
            #   t: vector of nodes 
            #   m: order of derivative sought 
            #   r: number of nodes to use from t 
            #   k: index of node whose weight is found

            if (m<0) || (m>r)        # undefined coeffs must be zero
                c = 0
            elseif (m==0) && (r==0)  # base case of one-point interpolation
                c = 1
            else                     # generic recursion
                if k<r
                    c = (t[r+1]*weight(t,m,r-1,k) -
                        m*weight(t,m-1,r-1,k))/(t[r+1]-t[k+1])
                else
                    numer = r > 1 ? prod(t[r]-x for x in t[1:r-1]) : 1
                    denom = r > 0 ? prod(t[r+1]-x for x in t[1:r]) : 1
                    β = numer/denom
                    c = β*(m*weight(t,m-1,r-1,r-1) - t[r]*weight(t,m,r-1,r-1))
                end
            end
            return c
        end
        r = length(t)-1
        w = zeros(size(t))
        return [ weight(t,m,r,k) for k=0:r ]
    end
end


# TODO Optimize - is there a variant which can @turbo the inner loop without 
# the additional buffers?
function fornberg_method_x!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3}, 
        weights::Vector{Vector{<:AbstractFloat}},
        stencils::Vector{UnitRange}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    fill!(res, zero(T))
    res_buffer = permutedims(res, [2,3,1])
    field_buffer = permutedims(field, [2,3,1])
    @inbounds @batch for i ∈ 1:nx
        w = weights[i]
        stencil = stencils[i]
        for is ∈ eachindex(stencil)
            h = stencil[is]
            @turbo for k ∈ 1:nz
                for j ∈ 1:ny
                    @inbounds res_buffer[j,k,i] += w[is]*field_buffer[j,k,h]
                end
            end
        end
    end
    permutedims!(res, res_buffer, [3,1,2])
    return nothing
end


function fornberg_method_y!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3}, 
        weights::Vector{Vector{<:AbstractFloat}},
        stencils::Vector{UnitRange}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    fill!(res, zero(T))
    @inbounds @batch for k ∈ 1:nz
        for j ∈ 1:ny
            w = weights[j]
            stencil = stencils[j]
            for is ∈ eachindex(stencil)
                h = stencil[is]
                @turbo for i ∈ 1:nx
                    @inbounds res[i,j,k] += w[is]*field[i,h,k]
                end
            end
        end
    end
    return nothing
end


function fornberg_method_z!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3}, 
        weights::Vector{Vector{<:AbstractFloat}},
        stencils::Vector{UnitRange}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    fill!(res, zero(T))
    @inbounds @batch for k ∈ 1:nz
        w = weights[k]
        stencil = stencils[k]
        for is ∈ eachindex(stencil)
            h = stencil[is]
            @turbo for j ∈ 1:ny
                for i ∈ 1:nx
                    @inbounds res[i,j,k] += w[is]*field[i,j,h]
                end
            end
        end
    end
    return nothing
end


function fornberg_method_1D!(
        res::AbstractArray{T,1},
        field::AbstractArray{T,1},
        weights::Vector{Vector{<:AbstractFloat}},
        stencils::Vector{UnitRange}
    ) where {T<:AbstractFloat}
    fill!(res, zero(T))
    @inbounds for i ∈ eachindex(field)
        w = weights[i]
        stencil = stencils[i]
        for is ∈ eachindex(stencil)
            h = stencil[is]
            res[i] += w[is]*field[h]
        end
    end
    return nothing
end


function get_stencils(
        n::Signed, stencil_size::Signed
    )::Vector{UnitRange}
    half = stencil_size ÷ 2
    stencils = Vector{UnitRange}(undef, n)
    for i ∈ eachindex(stencils)
        imin = max(1, i-half) # minimum index is 1
        imax = min(n, i+half) # maximum index is nx
        if imax - imin + 1 < stencil_size
            if i <= half
                imin, imax = 1, stencil_size
            else
                imin, imax = n-stencil_size+1, n
            end
        end
        stencils[i] = imin:imax
    end
    return stencils
end



end