module FDM

using Polyester
using LoopVectorization

export get_weights, get_stencils
export fornberg_method_1D!
export fornberg_method_2D_x!, fornberg_method_2D_y!
export fornberg_method_x!, fornberg_method_y!, fornberg_method_z!


let
    """
        get_weights(axis, stencil_size)
    Initialize the Fornberg weights along a complete axis with stencil size.
    """
    global function get_weights(
            axis::Vector{T}, stencil_size::Signed; order=1
        )::Matrix{T} where {T<:AbstractFloat}
        nx = length(axis)
        # res = Vector{Vector{<:AbstractFloat}}(undef, nx)
        res = Matrix{T}(undef, stencil_size, nx)
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
            res[:,i] .= fdweights(axis[imin:imax] .- axis[i], order)[:]
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


function _fornberg_method_x!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3},
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    res_buffer = permutedims(res, [2,3,1])
    field_buffer = permutedims(field, [2,3,1])
    @inbounds @batch for i ∈ 1:nx
        w = view(weights,:, i)
        stencil = view(stencils, :, i)
        for is ∈ eachindex(stencil)
            @turbo for k ∈ 1:nz
                for j ∈ 1:ny
                    @inbounds res_buffer[j,k,i] += w[is]*field_buffer[j,k,stencil[is]]
                end
            end
        end
    end
    permutedims!(res, res_buffer, [3,1,2])
    return nothing
end


# Optimized version of above
function fornberg_method_x!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3},
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    @inbounds @batch for k ∈ 1:nz
        for j ∈ 1:ny
            for i ∈ 1:nx
                w = view(weights, :, i)
                stencil = view(stencils, :, i)
                acc = zero(T)
                # NOTE: @turbo makes this loop ≈3x slower
                for is ∈ eachindex(stencil)
                    acc += w[is]*field[stencil[is],j,k]
                end
                res[i,j,k] = acc
            end
        end
    end
    return nothing
end


# Do not use this one - about 3x slower than the next version
function _fornberg_method_y!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3}, 
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    @inbounds @batch for k ∈ 1:nz
        for j ∈ 1:ny
            w = view(weights, :, j)
            stencil = view(stencils, :, j)
            for i ∈ 1:nx
                acc = zero(T)
                for is ∈ eachindex(stencil)        
                    acc += w[is]*field[i,stencil[is],k]
                end
                res[i,j,k] = acc
            end
        end
    end
    return nothing
end


# Benchmarks better than version above with accumulator (3x faster)
function fornberg_method_y!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3}, 
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    fill!(res, zero(T))
    nx, ny, nz = size(field)
    @inbounds @batch for k ∈ 1:nz
        for j ∈ 1:ny
            ws = view(weights, :, j)
            stencil = view(stencils, :, j)
            for is ∈ eachindex(stencil)
                w = ws[is]; h = stencil[is]
                @turbo for i ∈ 1:nx
                    res[i,j,k] += w*field[i,h,k]
                end
            end
        end
    end
    return nothing
end


function fornberg_method_z!(
        res::AbstractArray{T,3},
        field::AbstractArray{T,3},
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    fill!(res, zero(T))
    nx, ny, nz = size(field)
    @inbounds @batch for k ∈ 1:nz
        ws = view(weights, :, k)
        stencil = view(stencils, :, k)
        for is ∈ eachindex(stencil)
            w = ws[is]; h = stencil[is]
            for j ∈ 1:ny
                # NOTE: Better benchmark with @turbo at the innermost loop
                @turbo for i ∈ 1:nx
                    res[i,j,k] += w*field[i,j,h]
                end
            end
        end
    end
    return nothing
end


function fornberg_method_1D!(
        res::AbstractArray{T,1},
        field::AbstractArray{T,1},
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    fill!(res, zero(T))
    @inbounds for i ∈ eachindex(field)
        w = view(weights,: ,i)
        stencil = view(stencils, :, i)
        for is ∈ eachindex(stencil)
            res[i] += w[is]*field[stencil[is]]
        end
    end
    return nothing
end


function fornberg_method_2D_x!(
        res::AbstractArray{T,2},
        field::AbstractArray{T,2},
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    nx, ny = size(field)
    @inbounds @batch for j ∈ 1:ny
        for i ∈ 1:nx
            w = view(weights, :, i)
            stencil = view(stencils, :, i)
            acc = zero(T)
            # NOTE: @turbo makes this loop ≈3x slower
            for is ∈ eachindex(stencil)
                acc += w[is]*field[stencil[is],j]
            end
            res[i,j] = acc
        end
    end
    return nothing
end


function fornberg_method_2D_y!(
        res::AbstractArray{T,2},
        field::AbstractArray{T,2}, 
        weights::Matrix{T},
        stencils::Matrix{<:Signed}
    ) where {T<:AbstractFloat}
    fill!(res, zero(T))
    nx, ny = size(field)
    @inbounds @batch for j ∈ 1:ny
        ws = view(weights, :, j)
        stencil = view(stencils, :, j)
        for is ∈ eachindex(stencil)
            w = ws[is]; h = stencil[is]
            @turbo for i ∈ 1:nx
                res[i,j] += w*field[i,h]
            end
        end
    end
    return nothing
end


function get_stencils(
        n::Signed, stencil_size::Signed
    # )::Vector{UnitRange}
    )::Matrix{<:Signed}
    half = stencil_size ÷ 2
    # stencils = Vector{UnitRange}(undef, n)
    stencils = Matrix{Int}(undef, stencil_size, n)
    for i ∈ 1:n
        imin = max(1, i-half) # minimum index is 1
        imax = min(n, i+half) # maximum index is nx
        if imax - imin + 1 < stencil_size
            if i <= half
                imin, imax = 1, stencil_size
            else
                imin, imax = n-stencil_size+1, n
            end
        end
        # stencils[i] = imin:imax
        j = 1
        for is ∈ imin:imax
            stencils[j,i] = is
            j += 1
        end
    end
    return stencils
end



end