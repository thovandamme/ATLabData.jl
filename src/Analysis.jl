module Analysis

using ..DataStructures
using Polyester
using LoopVectorization

export gradient, gradient!, curl, curl!


let
verbose(type::String) = begin
    println("Calculating $type with $(Threads.nthreads()) threads.")
end


"""
    gradient(data)
"""
global function gradient(
        data::ScalarData{T,I}
    )::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose("gradient")
    res = Array{T}(undef, 3, data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        gradient3D!(res, data)
    else
        gradient2D!(res, data)
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
        data::ScalarData{T,I}
    )::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose("gradient")
    if data.grid.ny > 1
        gradient3D!(res.field, data)
    else
        gradient2D!(res.field, data)
    end
    res.name = "∇($(data.name))"
    return nothing
end


function gradient3D!(
        res::Array{T}, data::ScalarData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    @inbounds @batch for k ∈ eachindex(data.grid.z)
        for j ∈ eachindex(data.grid.y)
            for i ∈ eachindex(data.grid.x)
                # ∂/∂x
                if i==1
                    res[1,i,j,k] = data.field[i+1,j,k] - data.field[i,j,k]
                    res[1,i,j,k] = res[1,i,j,k]/(data.grid.x[i+1] - data.grid.x[i])
                elseif i==data.grid.nx
                    res[1,i,j,k] = data.field[i,j,k] - data.field[i-1,j,k]
                    res[1,i,j,k] = res[1,i,j,k]/(data.grid.x[i] - data.grid.x[i-1])
                else
                    res[1,i,j,k] = data.field[i+1,j,k] - data.field[i-1,j,k]
                    res[1,i,j,k] = res[1,i,j,k]/(data.grid.x[i+1] - data.grid.x[i-1])
                end
                # ∂/∂y
                if j==1
                    res[2,i,j,k] = data.field[i,j+1,k] - data.field[i,j,k]
                    res[2,i,j,k] = res[2,i,j,k]/(data.grid.y[j+1] - data.grid.y[j])
                elseif j==data.grid.ny
                    res[2,i,j,k] = data.field[i,j,k] - data.field[i,j-1,k]
                    res[2,i,j,k] = res[2,i,j,k]/(data.grid.y[j] - data.grid.y[j-1])
                else
                    res[2,i,j,k] = data.field[i,j+1,k] - data.field[i,j-1,k]
                    res[2,i,j,k] = res[2,i,j,k]/(data.grid.y[j+1] - data.grid.y[j-1])
                end
                # ∂/∂z
                if k==1
                    res[3,i,j,k] = data.field[i,j,k+1] - data.field[i,j,k]
                    res[3,i,j,k] = res[2,i,j,k]/(data.grid.z[k+1] - data.grid.z[k])
                elseif k==data.grid.nz
                    res[3,i,j,k] = data.field[i,j,k] - data.field[i,j,k-1]
                    res[3,i,j,k] = res[3,i,j,k]/(data.grid.z[k] - data.grid.z[k-1])
                else
                    res[3,i,j,k] = data.field[i,j,k+1] - data.field[i,j,k-1]
                    res[3,i,j,k] = res[2,i,j,k]/(data.grid.z[k+1] - data.grid.z[k-1])
                end
            end
        end
    end
    return nothing
end


function gradient2D!(
        res::Array{T}, data::ScalarData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    @inbounds @batch for k ∈ eachindex(data.grid.z)
        for i ∈ eachindex(data.grid.x)
            # ∂/∂x
            if i==1
                res[1,i,1,k] = data.field[i+1,1,k] - data.field[i,1,k]
                res[1,i,1,k] = res[1,i,1,k]/(data.grid.x[i+1] - data.grid.x[i])
            elseif i==data.grid.nx
                res[1,i,1,k] = data.field[i,1,k] - data.field[i-1,1,k]
                res[1,i,1,k] = res[1,i,1,k]/(data.grid.x[i] - data.grid.x[i-1])
            else
                res[1,i,1,k] = data.field[i+1,1,k] - data.field[i-1,1,k]
                res[1,i,1,k] = res[1,i,1,k]/(data.grid.x[i+1] - data.grid.x[i-1])
            end
            # ∂/∂z
            if k==1
                res[3,i,1,k] = data.field[i,1,k+1] - data.field[i,1,k]
                res[3,i,1,k] = res[2,i,1,k]/(data.grid.z[k+1] - data.grid.z[k])
            elseif k==data.grid.nz
                res[3,i,1,k] = data.field[i,1,k] - data.field[i,1,k-1]
                res[3,i,1,k] = res[3,i,1,k]/(data.grid.z[k] - data.grid.z[k-1])
            else
                res[3,i,1,k] = data.field[i,1,k+1] - data.field[i,1,k-1]
                res[3,i,1,k] = res[2,i,1,k]/(data.grid.z[k+1] - data.grid.z[k-1])
            end
        end
    end
    return nothing
end


"""
    curl(data)
"""
global function curl(data::VectorData{T,I})::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose("curl")
    res = Array{T}(undef, 3, data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        curl3D!(res, data)
    else
        curl2D!(res, data)
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
Mutating variant fo curl(data). Stores result in preallocated VectorData res.
"""
global function curl!(
        res::VectorData{T,I}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    verbose("curl")
    if data.grid.ny > 1
        curl3D!(res.field, data)
    else
        curl2D!(res.field, data)
    end
    res.name = "∇×($(data.name))"
    return nothing
end


function curl3D!(
        res::Array{T}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    # Setting the boundaries to zero for simplicity
    res[:,1,:,:] .= 0.0
    res[:,:,1,:] .= 0.0
    res[:,:,:,1] .= 0.0
    @inbounds @batch for k ∈ 2:data.grid.nz-1
        inv∂z = inv(data.grid.z[k+1] - data.grid.z[k-1])
        for j ∈ 2:data.grid.ny-1
            inv∂y = inv(data.grid.y[j+1] - data.grid.y[j-1])
            @turbo for i ∈ 2:data.grid.nx-1
                inv∂x = inv(data.grid.x[i+1] - data.grid.x[i-1])

                ∂1 = data.field[3,i,j+1,k] - data.field[3,i,j-1,k]
                ∂2 = data.field[2,i,j,k+1] - data.field[2,i,j,k-1]
                res[1,i,j,k] = ∂1*inv∂y - ∂2*inv∂z

                ∂1 = data.field[1,i,j,k+1] - data.field[1,i,j,k-1]
                ∂2 = data.field[3,i+1,j,k] - data.field[3,i-1,j,k]
                res[2,i,j,k] = ∂1*inv∂z - ∂2*inv∂x

                ∂1 = data.field[2,i+1,j,k] - data.field[2,i-1,j,k]
                ∂2 = data.field[1,i,j+1,k] - data.field[1,i,j-1,k]
                res[3,i,j,k] = ∂1*inv∂x - ∂2*inv∂y
            end
        end
    end
end


# This method is much slower
function _curl3D!(
        res::Array{T}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed} 
    @inline function ∂(fieldk, dir, i, j, k)
        dx = data.grid.x[i+1] - data.grid.x[i-1]
        dy = data.grid.y[j+1] - data.grid.y[j-1]
        dz = data.grid.z[k+1] - data.grid.z[k-1]
        if dir == 1
            return (fieldk[i+1,j,k] - fieldk[i-1,j,k]) / dx
        elseif dir == 2
            return (fieldk[i,j+1,k] - fieldk[i,j-1,k]) / dy
        elseif dir == 3
            return (fieldk[i,j,k+1] - fieldk[i,j,k-1]) / dz
        else
            error("field does not have this component")
        end
    end 
    @inbounds @batch for k ∈ 2:data.grid.nz-1
        for j ∈ 2:data.grid.ny-1
            for i ∈ 2:data.grid.nx-1
                for α ∈ 1:3 # curl index
                    for β ∈ 1:3 # derivate index
                        for γ ∈ 1:3 # field index 
                            ε = levi_civita(α, β, γ)
                            ε==0 && continue
                            res[α,i,j,k] += ε*∂(
                                view(data.field, γ, :, :, :), β, i, j, k
                            )
                        end
                    end
                end
            end
        end
    end
    return nothing
end


@inline function levi_civita(i::Int, j::Int, k::Int)
    if i == j || j == k || i == k
        return 0
    elseif (i, j, k) in ((1,2,3), (2,3,1), (3,1,2))
        return 1
    else
        return -1
    end
end


function curl2D!(
        res::Array{T}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    res .= 0.0
    @inbounds @batch for k ∈ 2:data.grid.nz-1
        inv∂z = inv(data.grid.z[k+1] - data.grid.z[k-1])
        @turbo for i ∈ 2:data.grid.nx-1
            inv∂x = inv(data.grid.x[i+1] - data.grid.x[i-1])
            ∂1 = data.field[1,i,1,k+1] - data.field[1,i,1,k-1]
            ∂2 = data.field[3,i+1,1,k] - data.field[3,i-1,1,k]
            res[2,i,1,k] = ∂1*inv∂z - ∂2*inv∂x
        end
    end
end


global function divergence(
        data::VectorData{T,I}
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose("divergence")
    res = Array{T}(undef, 3, data.grid.nx, data.grid.ny, data.grid.nz)
    if data.grid.ny > 1
        divergence3D!(res, data)
    else
        divergence2D!(res, data)
    end
    return ScalarData(
        name = "∇⋅($(data.name))",
        grid = data.grid,
        time = data.time,
        field = res
    )
end


global function divergence!(
        res::ScalarData{T,I}, data::VectorData{T,I}
    )::ScalarData{T,I} where {T<:AbstractFloat, I<:Signed}
    verbose("divergence")
    if data.grid.ny > 1
        divergence3D!(res.field, data)
    else
        divergence2D!(res.field, data)
    end
    res.name = "∇⋅($(data.name))"
    return nothing
end


function divergence3D!(
        res::Array{T}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    res .= 0.0
    @inbounds @batch for k ∈ 2:data.grid.nz-1
        inv∂z = inv(data.grid.z[k+1] - data.grid.z[k-1])
        for j ∈ 2:data.grid.ny-1
            inv∂y = inv(data.grid.y[j+1] - datag.rid.y[j-1])
            for i ∈ 2:data.grid.nx-1
                inv∂x = inv(data.grid.x[i+1] - data.grid.x[i-1])
                res[i,j,k] = (data.field[1,i+1,j,k] - data.field[1,i-1,j,k])*inv∂x
                res[i,j,k] += (data.field[2,i,j+1,k] - data.field[2,i,j-1,k])*inv∂y
                res[i,j,k] += (data.field[3,i,j,k+1] - data.field[3,i,j,k-1])*inv∂z
            end
        end
    end
end


function divergence2D!(
        res::Array{T}, data::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    res .= 0.0
    @inbounds @batch for k ∈ 2:data.grid.nz-1
        inv∂z = inv(data.grid.z[k+1] - data.grid.z[k-1])
        for i ∈ 2:data.grid.nx-1
            inv∂x = inv(data.grid.x[i+1] - data.grid.x[i-1])
            res[i,j,k] = (data.field[1,i+1,j,k] - data.field[1,i-1,j,k])*inv∂x    
            res[i,j,k] += (data.field[3,i,j,k+1] - data.field[3,i,j,k-1])*inv∂z
        end
    end
end
end # end of let scope    


end