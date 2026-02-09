module Analysis

using ..DataStructures
using Base.Threads

export gradient, curl


let
"""
    gradient(data; order=4)
Return the gradient of _data_ by using the packages _FiniteDifferences_  and 
_Iterpolations_.  
_order_ determines the numerical error order for the derivatives.
"""
global function gradient(
        data::ScalarData{T,I}
    )::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
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
        xfield = res[1,:,:,:],
        yfield = res[2,:,:,:],
        zfield = res[3,:,:,:]
    )
end


# global function gradient!(
#         res::VectorData{T,I},
#         data::ScalarData{T,I}
#     )::VectorData{T,I} where {T<:AbstractFloat, I<:Signed}
#     if data.grid.ny > 1
#         gradient3D!(res.field, data)
#     else
#         gradient2D!(res.field, data)
#     end
#     return nothing
# end


function gradien3D!(
        res::Array{T}, data::ScalarData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    @inbounds @threads for k ∈ eachindex(data.grid.z)
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


function gradien2D!(
        res::Array{T}, data::ScalarData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    @inbounds @threads for k ∈ eachindex(data.grid.z)
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

end # end of let scope for gradient


function curl(data::VectorData; order::Int=4)::VectorData
    # TODO: 3D, faster?, parallise
    
    println("Calculating curl ...")
    printstyled("    "*data.name*"\n", color=:cyan)
    
    xres = zeros(Float32, (data.grid.nx, data.grid.ny, data.grid.nz))
    yres = zeros(Float32, (data.grid.nx, data.grid.ny, data.grid.nz))
    zres = zeros(Float32, (data.grid.nx, data.grid.ny, data.grid.nz))
    
    if data.grid.nz > 1
        kmin = order÷2 + 1; kmax = data.grid.nz - order - 1
    else
        kmin = 1; kmax = data.grid.nz
    end
    if data.grid.ny > 1
        jmin = order÷2 + 1; jmax = data.grid.ny - order - 1
    else
        jmin = 1; jmax = data.grid.ny
    end
    if data.grid.nx > 1
        imin = order÷2 + 1; imax = data.grid.nx - order - 1
    else
        imin = 1; imax = data.grid.nx
    end

    """ Not the real curl: computes curl for each y-slice (2D) ... TODO """
    for j ∈ jmin:jmax
        xitp = interpolate(
            (data.grid.x, data.grid.z), # Nodes of the grid
            data.xfield[:,j,:], # Field to be interpolated
            Gridded(Linear())
        )
        # yitp = interpolate(
        #     (data.grid.x, data.grid.y), # Nodes of the grid
        #     data.yfield[:,:,k], # Field to be interpolated
        #     Gridded(Linear())
        # )
        zitp = interpolate(
            (data.grid.x, data.grid.z), # Nodes of the grid
            data.zfield[:,j,:], # Field to be interpolated
            Gridded(Linear())
        )
        for k ∈ kmin:kmax
            for i ∈ imin:imax
                xgrad = grad(central_fdm(order, 1), xitp, data.grid.x[i], data.grid.z[k])
                # ygrad = grad(central_fdm(order, 1), yitp, data.grid.x[i], data.grid.z[k])
                zgrad = grad(central_fdm(order, 1), zitp, data.grid.x[i], data.grid.z[k])

                # xres[i,j,k] = zgrad[2] - ygrad[3]
                yres[i,j,k] = xgrad[3] - zgrad[1]
                # zres[i,j,k] = ygrad[1] - xgrad[2]
            end
        end
    end
    return VectorData(
        name="curl($(data.name))", time=data.time, grid=data.grid,
        xfield=xres, yfield=yres, zfield=zres
    )
end

    
end