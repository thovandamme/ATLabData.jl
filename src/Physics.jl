module Physics

using Polyester, LoopVectorization
using ..DataStructures, ..IO, ..Basics, ..Statistics, ..Calculus

export vorticity, enstrophy, Ri

export kinenergy, kinenergy!
export Reynolds_stress_tensor, Reynolds_stress_tensor!
export dissipation_rate, dissipation_rate!
export dissipation_tensor, dissipation_tensor!
export production_rate!


function do_verbose(field::String)
    println("Calculating $field with $(Threads.nthreads()) threads.")
    return nothing
end


"""
    vorticity(data) -> VectorData
Return the curl of data, thus is a physical alternatice to curl if data 
is a velocity field.

    vorticity(dir, time) -> VectorData
Looks for the proper velocity files in dir that are nearest to _time_ and 
computes the curl.
"""
vorticity(u::VectorData)::VectorData = curl(u)
vorticity(dir::String, time::Real)::VectorData = curl(load(
    file_for_time(dir, "VelocityVector", time, ".1"),
    file_for_time(dir, "VelocityVector", time, ".2"),
    file_for_time(dir, "VelocityVector", time, ".3")
))


"""
    enstrophy(u) -> ScalarData
Calculates the enstrophy of the givem velocity field.

    enstrophy(dir, time) -> ScalarData
Looks in _dir_ for the velocity field at _time_ and calculates the appropriate 
    enstrophy.
"""
enstrophy(u::VectorData)::ScalarData = ScalarData(
    name = "enstrophy(" * u.name * ")", 
    time = u.time, 
    grid = u.grid, 
    field = norm(vorticity(u)).field.^2
)
enstrophy(dir::String, time::Real)::ScalarData = enstrophy(load(
    file_for_time(dir, "VelocityVector", time, ".1"),
    file_for_time(dir, "VelocityVector", time, ".2"),
    file_for_time(dir, "VelocityVector", time, ".3")
))


"""
Computes and returns the local Richardson number field from the given 
buoyancy and velocity fields.

    Ri(b, u) -> ScalarData
_u_ is given as _VectorData_.

    RI(b, ux, uy, uz) -> ScalarData
The single components are given as _Data_.

    Ri(dir, time) -> ScalarData
Looks in _dir_ for the buoyancy and velocity fields for _time_.
"""
Ri(b::ScalarData, u::VectorData)::ScalarData = ScalarData(
    name = "Rig("*u.name*")",
    time = b.time,
    grid = b.grid,
    field = norm(gradient(u)).field.^2 ./ norm(gradient(b))
)
Ri(b::ScalarData, ux::ScalarData, uy::ScalarData, uz::ScalarData)::ScalarData = ScalarData(
    name = "Rig("*b.name*")",
    time = b.time,
    grid = b.grid,
    field = (
        norm(gradient(b)).field ./ (norm(gradient(ux)).field.^2 
        + norm(gradient(uy)).field.^2 + norm(gradient(uz)).field.^2)
    )
)
Ri(dir::String, time::Real)::ScalarData = Ri(
    load(dir, "Buoyancy", time),
    load(dir, "VelocityVector", time, ".1"),
    load(dir, "VelocityVector", time, ".2"),
    load(dir, "VelocityVector", time, ".3"),
)


function tke(u::VectorData)::ScalarData
    buffer = flucs(u)
    return ScalarData(
        name = "tke($(u.name))", 
        grid = u.grid,
        iteration = u.iteration, 
        time = u.time,
        field = 0.5 .* (buffer.field[1,:,:,:].^2 .+ buffer.field[2,:,:,:].^2 .+ buffer.field[3,:,:,:].^2)
    )
end


function turbulent_diffusivity(
        flux::AveragesData, 
        mean::AveragesData;
        axis::Vector{<:AbstractFloat} = mean.grid.z
    )::AveragesData
    return AveragesData(
        name = "turbDiff($(flux.name))",
        time = flux.time,
        z = axis,
        field = turbulent_diffusivity(flux.field, mean.field, axis=axis)
    )
end


function turbulent_diffusivity(
        flux::Vector{T}, mean::Vector{T}; axis::Vector{T}
    )::Vector{T} where {T<:AbstractFloat}
    return - flux ./ ∂x(mean, axis)
end


################################################################################
#                       Kinetic energy statistic
################################################################################
function kinenergy!(
        res::Array{T,1}, u::AbstractArray{T,4}
    ) where {T<:AbstractFloat}
    nv, nx, ny, nz = size(u)
    @inbounds @batch for k ∈ 1:nz
        acc = zero(T)
        @turbo for j ∈ 1:ny, i ∈ 1:nx
            for h ∈ 1:nv
                acc += u[h,i,j,k]^2
            end
        end
        res[k] = acc/(2*nx*ny)
    end
    return nothing
end


function kinenergy!(
        res::Array{T,1}, u::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    kinenergy!(res, u.field)
    return nothing
end


function kinenergy(u::VectorData{T,I})::Array{T,1} where {T<:AbstractFloat, I<:Signed}
    res = similar(u.grid.z)
    kinenergy!(res, u.field)
    return res
end


################################################################################
#                       Reynolds stress tensor
################################################################################
function Reynolds_stress_tensor!(
        res::Array{T,3}, field::Array{T,4}; verbose=true
    ) where {T<:AbstractFloat}
    verbose && do_verbose("Rᵢⱼ")
    nv, nx, ny, nz = size(field)
    fill!(res, zero(T))
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            for g ∈ 1:nv
                acc = zero(T)
                for j ∈ 1:ny
                    @turbo for i ∈ 1:nx
                        @inbounds acc += field[g,i,j,k]*field[h,i,j,k]
                    end
                end
                res[g,h,k] = acc/(nx*ny)
            end
        end
    end
    return nothing
end


function Reynolds_stress_tensor(
        field::Array{T,4}
    )::Array{T,3} where {T<:AbstractFloat}
    res = Array{T, 3}(undef, 3, 3, size(field)[4])
    Reynolds_stress_tensor!(res, field)
    return res
end


function Reynolds_stress_tensor(
        data::VectorData{T,I}
    )::Array{T,3} where {T<:AbstractFloat, I<:Signed}
    res = Array{T, 3}(undef, 3, 3, size(data.field)[4])
    Reynolds_stress_tensor!(res, field.field)
    return res
end


################################################################################
#                           Turbulence dissipation
################################################################################
function dissipation_tensor!(
        res::AbstractArray{T,3}, ∇u::AbstractArray{T,5}, Re::Real; verbose=true
    ) where {T<:AbstractFloat}
    # ∇u is the jacobian of the vector-valued velocity
    # NOTE: @turbo leads here to much more allocations and longer running time
    verbose && do_verbose("εᵢⱼ")
    nv, nx, ny, nz = size(∇u[1,:,:,:,:])
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            for g ∈ 1:nv
                acc = zero(T)
                for j ∈ 1:ny
                    for i ∈ 1:nx
                        for f ∈ 1:nv
                            @inbounds acc += ∇u[f,h,i,j,k]*∇u[f,g,i,j,k]
                        end
                    end
                end
                res[g,h,k] = 2*Re^(-1)*acc/(nx*ny)
            end
        end
    end
    return nothing 
end


function dissipation_tensor(
        ∇u::AbstractArray{T,5}, Re::Real; verbose=true
    ) where {T<:AbstractFloat}
    res = Array{T,3}(undef, 3, 3, size(∇u)[end])
    dissipation_tensor!(res, ∇u, Re, verbose=verbose)
    return res
end


function dissipation_tensor(
        u::AbstractArray{T,4}, grid::Grid{T,I}, Re::Real; verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    ∇u = jacobian(u, grid, verbose=false)
    return dissipation_tensor(∇u, Re, verbose=verbose)
end


function dissipation_tensor(
        u::VectorData{T,I}; verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    return dissipation_tensor(u.field, u.grid, Re, verbose=verbose)
end


function dissipation_rate!(
        res::AbstractArray{T,1}, E::AbstractArray{T,3}; verbose=true
    ) where {T<:AbstractFloat}
    verbose && do_verbose("ε")
    for k ∈ eachindex(res)
        res[k] = 0.5*(E[1,1,k] + E[2,2,k] + E[3,3,k])
    end
    return nothing
end


function dissipation_rate!(
        res::AbstractArray{T,1}, ∇u::AbstractArray{T,5}, Re::Real; verbose=true
    ) where {T<:AbstractFloat}
    # This is much more efficient than utilizing the dissipation as above
    # However, if εᵢⱼ is calculated anyway, above is better
    verbose && do_verbose("ε")
    nv, nx, ny, nz = size(∇u)[2:end]
    @inbounds @batch for k ∈ 1:nz
        acc = zero(T)
        for j ∈ 1:ny
            for i ∈ 1:nx
                S = view(∇u, :, :, i, j, k)
                for h ∈ 1:nv
                    for g ∈ 1:nv
                        acc += (0.5*(S[g,h] + S[h,g]))^2
                    end
                end
            end
        end
        res[k] = 2*Re^(-1)*acc/(nx*ny)
    end
    return nothing
end


function dissipation_rate!(
        res::AbstractArray{T,1}, u::AbstractArray{T,4}, grid::Grid{T,I}, Re::Real; verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    ∇u = jacobian(u, grid, verbose=false)
    dissipation_rate!(res, ∇u, Re, verbose=verbose)
    return nothing
end


function dissipation_rate(
        ∇u::AbstractArray{T,5}, Re::Real; verbose=true
    ) where {T<:AbstractFloat}
    res = Array{T,1}(undef, size(∇u)[end])
    dissipation_rate!(res, ∇u, Re, verbose=verbose)
    return res
end


function dissipation_rate(
        u::AbstractArray{T,4}, grid::Grid{T,I}, Re::Real; verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    ∇u = jacobian(u, grid, verbose=false)
    return dissipation_rate(∇u, Re, verbose=verbose)
end


function dissipation_rate(
        u::VectorData{T,I}, Re::Real; verbose=true
    ) where {T<:AbstractFloat, I<:Signed}
    return dissipation_rate(u.field, u.grid, Re, verbose=verbose)
end


################################################################################
#                           Turbulence shear production
################################################################################
function production_tensor!()
    return nothing
end


function production_rate!(
        res::AbstractArray{T,1}, R::AbstractArray{T,3}, S::AbstractArray{T,3}
    ) where {T<:AbstractFloat}
    for k ∈ eachindex(res)
        acc = zero(T)
        for h ∈ 1:3, g ∈ 1:3
            acc += R[g,h,k]*S[g,h,k]
        end
        res[k] = -acc
    end
    return nothing
end


function production_rate!(
        res::AbstractArray{T,1}, R::AbstractArray{T,3}, ∇u::AbstractArray{T,5}
    ) where {T<:AbstractFloat}
    # ∇u has to be the jacobian of the total velcoity field
    # ⟨sᵢⱼ⟩(z) with i=h and j=g (sᵢⱼ from the total field, not only fluctuations)
    S = similar(R) # ⟨sᵢⱼ⟩(z)
    nv, nx, ny, nz = size(∇u)[2:end]
    println("Calculating ⟨sᵢⱼ⟩(z).")
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            for g ∈ 1:nv
                acc = zero(T)
                pointer1 = view(∇u, g, h, :, :, k)
                pointer2 = view(∇u, h, g, :, :, k)
                @turbo for j ∈ 1:ny
                    for i ∈ 1:nx
                        @inbounds acc += 0.5*(pointer1[i,j] + pointer2[i,j])
                    end
                end
                S[g,h,k] = acc/(fulldata.grid.nx*fulldata.grid.ny)
            end
        end
    end
    production_rate!(res, R, S)
    return nothing
end


function production_rate!(
        res::AbstractArray{T,1}, u::AbstractArray{T,4}, grid::Grid{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    ∇u = jacobian(u, grid)
    production_rate!(res, R, ∇u)
    return nothing
end


function production_rate!(
        res::AbstractArray{T,1}, u::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    production_rate!(res, u.field, u.grid)
    return nothing
end


# TODO Allocating variants


end