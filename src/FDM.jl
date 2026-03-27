module FDM

using Polyester

export weights, get_stencils
export fornberg_method!, fornberg_method_x!, fornberg_method_y!, fornberg_method_z!


let
    """
        weights(axis, stencil_size)
    Initialize the Fornberg weights along a complete axis with stencil size.
    """
    global function weights(
            axis::Vector{T}, stencil_size::Signed
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
            res[i] = fdweights(axis[imin:imax] .- axis[i], 1)
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


# Applies differentiation in all three dimensions within one loop to 
# avoid multi-threading overhead compared to calling fornberg_method_x/y/z 
# separately. Usefull for gradient computation.
function fornberg_method!(
        resx::Array{T,3},
        resy::Array{T,3},
        resz::Array{T,3},
        field::Array{T,3},
        weights_x::Vector{Vector{<:AbstractFloat}},
        weights_y::Vector{Vector{<:AbstractFloat}},
        weights_z::Vector{Vector{<:AbstractFloat}},
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    x_stencils = get_stencils(nx, stencil_size)
    y_stencils = get_stencils(ny, stencil_size)
    z_stencils = get_stencils(nz, stencil_size)
    @inbounds @batch for k ∈ 1:nz
        for j ∈ 1:ny
            for i ∈ 1:nx
                resx[i,j,k] = sum(weights_x[i] .* field[x_stencils[i],j,k])
                resy[i,j,k] = sum(weights_y[i] .* field[i,y_stencils[j],k])
                resz[i,j,k] = sum(weights_z[i] .* field[i,j,z_stencils[k]])
            end
        end
    end
    return nothing
end


function fornberg_method_x!(
        res::Array{T,3},
        field::Array{T,3}, 
        weights::Vector{Vector{<:AbstractFloat}},
        stencil_size::Signed
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    stencils = get_stencils(nx, stencil_size)
    @inbounds @batch for k ∈ eachindex(1:nz)
        for j ∈ eachindex(1:ny)
            for i ∈ eachindex(1:nx)
                res[i,j,k] = sum(weights[i] .* field[stencils[i],j,k])
            end
        end
    end
    return nothing
end


function fornberg_method_y!(
        res::Array{T,3},
        field::Array{T,3}, 
        weights::Vector{Vector{<:AbstractFloat}},
        stencil_size::Signed
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    stencils = get_stencils(ny, stencil_size)
    @inbounds @batch for k ∈ eachindex(1:nz)
        for j ∈ eachindex(1:ny)
            for i in eachindex(1:nx)
                res[i,j,k] = sum(weights[j] .* field[i,stencils[j],k])
            end
        end
    end
    return nothing
end


function fornberg_method_z!(
        res::Array{T,3},
        field::Array{T,3}, 
        weights::Vector{Vector{<:AbstractFloat}},
        stencil_size::Signed
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(field)
    stencils = get_stencils(nz, stencil_size)
    @inbounds @batch for k ∈ eachindex(1:nz)
        for j ∈ eachindex(1:ny)
            for i in eachindex(1:nx)
                res[i,j,k] = sum(weights[k] .* field[i,j,stencils[k]])
            end
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