module Physics

using Polyester, LoopVectorization
using ..DataStructures, ..IO, ..Basics, ..Statistics, ..ArrayCalculus

export kinenergy, kinenergy!
export Reynolds_stress_tensor, Reynolds_stress_tensor!
export dissipation_rate, dissipation_rate!
export dissipation_tensor, dissipation_tensor!
export production_rate!, production_rate
export vorticity!, vorticity
export buoyancy_flux!
export triple_velocity_correlation_vector!, triple_velocity_correlation_vector
export triple_velocity_correlation_transport!, triple_velocity_correlation_transport
export pressure_veclocity_corrrelation_vector!, pressure_veclocity_corrrelation_vector
# export pressure_veclocity_corrrelation_transport!, pressure_veclocity_corrrelation_transport
export visc_stress_work_vector!, visc_stress_work_vector
export visc_stress_work_transport!, visc_stress_work_transport
export turbulent_flux_vector!, turbulent_flux_vector
export turbulent_transport!, turbulent_transport


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
        res::AbstractArray{T,1}, u::AbstractArray{T,4}
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
        res::AbstractArray{T,1}, u::VectorData{T,I}
    ) where {T<:AbstractFloat, I<:Signed}
    kinenergy!(res, u.field)
    return nothing
end


function kinenergy!(
        res::AbstractArray{T,1}, R::AbstractArray{T,3}
    ) where {T<:AbstractFloat}
    @turbo for k ∈ eachindex(res)
        res[k] = 0.5*(R[1,1,k] + R[2,2,k] + R[3,3,k])
    end
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
# TODO OPtimize: Rᵢⱼ is symmetric
function Reynolds_stress_tensor!(
        res::Array{T,3}, field::AbstractArray{T,4}; verbose=true
    ) where {T<:AbstractFloat}
    verbose && do_verbose("Rᵢⱼ")
    nv, nx, ny, nz = size(field)
    fill!(res, zero(T))
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            for g ∈ 1:nv
                acc = zero(T)
                @turbo for j ∈ 1:ny, i ∈ 1:nx
                    @inbounds acc += field[g,i,j,k]*field[h,i,j,k]
                end
                res[g,h,k] = acc/(nx*ny)
            end
        end
    end
    return nothing
end


function Reynolds_stress_tensor(
        field::AbstractArray{T,4}
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
    nv, nx, ny, nz = size(∇u)[2:end]
    @inbounds @batch for k ∈ 1:nz
        acc = zero(T)
        for h ∈ 1:nv
            for g ∈ 1:nv
                acc2 = zero(T)
                pointer1 = view(∇u, g, h, :, :, k)
                pointer2 = view(∇u, h, g, :, :, k)
                @turbo for j ∈ 1:ny
                    for i ∈ 1:nx
                        @inbounds acc2 += 0.5*(pointer1[i,j] + pointer2[i,j])
                    end
                end
                # S[g,h,k] = acc2/(nx*ny)
                acc = R[g,h,k]*acc2/(nx*ny)
            end
        end
        res[k] = -acc
    end
    return nothing
end


# function production_rate!(
#         res::AbstractArray{T,1}, R::AbstractArray{T,3}, ∇u::AbstractArray{T,5}
#     ) where {T<:AbstractFloat}
#     # ∇u has to be the jacobian of the total velcoity field
#     # ⟨sᵢⱼ⟩(z) with i=h and j=g (sᵢⱼ from the total field, not only fluctuations)
#     S = similar(R) # ⟨sᵢⱼ⟩(z)
#     nv, nx, ny, nz = size(∇u)[2:end]
#     # println("Calculating ⟨sᵢⱼ⟩(z).")
#     @inbounds @batch for k ∈ 1:nz
#         for h ∈ 1:nv
#             for g ∈ 1:nv
#                 acc = zero(T)
#                 pointer1 = view(∇u, g, h, :, :, k)
#                 pointer2 = view(∇u, h, g, :, :, k)
#                 @turbo for j ∈ 1:ny
#                     for i ∈ 1:nx
#                         @inbounds acc += 0.5*(pointer1[i,j] + pointer2[i,j])
#                     end
#                 end
#                 S[g,h,k] = acc/(nx*ny)
#             end
#         end
#     end
#     production_rate!(res, R, S)
#     return nothing
# end


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


################################################################################
#                           Vorticity
################################################################################

function vorticity!(
        res::Array{T,4}, ∇u::AbstractArray{T,5}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(res)[2:end]
    @tturbo for k ∈ 1:nz, j ∈ 1:ny, i ∈ 1:nx
        res[1,i,j,k] = ∇u[2,3,i,j,k] - ∇u[3,2,i,j,k]
        res[2,i,j,k] = ∇u[3,1,i,j,k] - ∇u[1,3,i,j,k]
        res[3,i,j,k] = ∇u[1,2,i,j,k] - ∇u[2,1,i,j,k]
    end
    return nothing
end


function vorticity(∇u::AbstractArray{T,5})::Array{T,4} where {T<:AbstractFloat}
    nx, ny, nz = size(∇u)[2:end]
    res = Array{T,4}(undef, 3, nx, ny, nz)
    vorticity!(res, ∇u)
    return res
end


################################################################################
#                 Buoyancy production rate / Buoyancy flux
################################################################################

function buoyancy_flux!(
        res::AbstractArray{T,1}, b::AbstractArray{T,3}, w::AbstractArray{T,3}
    ) where {T<:AbstractFloat}
    nx, ny, nz = size(b)
    @inbounds @batch for k ∈ 1:nz
        acc = zero(T)
        @turbo for j ∈ 1:ny, i ∈ 1:nx
            acc += b[i,j,k]*w[i,j,k]
        end
        res[k] = acc/(nx*ny)
    end
    return nothing
end


################################################################################
#                       Turbulent transport / flux
################################################################################

# ∂ⱼTⱼ = ∂₃T₃ (mean in flux vector is over index 1 and 2, hence ∂₁T₁=∂₂T₂=0)
function turbulent_transport!(
        res::AbstractVector{T}, 
        u::AbstractArray{T,4}, ∇u::AbstractArray{T,5}, 
        Re::Real, ρ₀::Real, z::AbstractVector{T}
    ) where {T<:AbstractFloat}
    Tⱼ = turbulent_flux_vector(u, ∇u, Re, ρ₀)
    res .= ∂1(view(Tⱼ, 3, :), z)
    # @turbo for k ∈ eachindex(res)
    #     @inbounds res[k] = ∂₁T₁[k] + ∂₂T₂[k] + ∂₃T₃[k]
    # end
    return nothing
end


function turbulent_transport(
        u::AbstractArray{T,4}, ∇u::AbstractArray{T,5}, 
        Re::Real, ρ₀::Real, z::AbstractVector{T}
    )::Vector{T} where {T<:AbstractFloat}
    res = Vector{T}(undef, size(u)[end])
    turbulent_transport!(res, u, ∇u, Re, ρ₀, z)
    return res
end


# Tⱼ = ρ₀⟨uⱼ(uᵢ)²/2⟩ + ⟨puⱼ⟩ + ⟨τᵢⱼuᵢ⟩
function turbulent_flux_vector!(
        res::AbstractArray{T,2}, 
        u::AbstractArray{T,4}, ∇u::AbstractArray{T,5}, Re::Real, ρ₀::Real
    ) where {T<:AbstractFloat}
    buffer = similar(res)
    triple_velocity_correlation_vector!(res, ρ₀, u)
    # pressure_veclocity_corrrelation_vector!(buffer, p, u)
    # res .+= buffer
    visc_stress_work_vector!(buffer, ∇u, u, Re)
    res .+= buffer
    return nothing
end


function turbulent_flux_vector(
        u::AbstractArray{T,4}, ∇u::AbstractArray{T,5}, Re::Real, ρ₀::Real
    )::Array{T,2} where {T<:AbstractFloat}
    nv = size(u)[1]; nz = size(u)[end]
    res = Array{T,2}(undef, nv ,nz)
    turbulent_flux_vector!(res, u, ∇u, Re, ρ₀)
    return res
end


# ρ₀⟨uⱼ(uᵢ)²/2⟩
function triple_velocity_correlation_vector!(
        res::AbstractArray{T, 2}, ρ₀::Real, u::AbstractArray{T, 4}
    ) where {T<:AbstractFloat}
    nv, nx, ny, nz = size(u)
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            acc = zero(T) # Mean over i and j
            for j ∈ 1:ny
                for i ∈ 1:nx
                    acc2 = zero(T)
                    for g ∈ 1:nv
                        acc2 += u[g,i,j,k]*u[g,i,j,k]
                    end
                    acc += acc2/2*u[h,i,j,k]
                end
            end
            res[h,k] = ρ₀*acc/T(nx*ny)
        end
    end
    return nothing
end


function triple_velocity_correlation_vector(
        ρ₀::Real, u::AbstractArray{T, 4}
    )::Array{T,2} where {T<:AbstractFloat}
    nv, nx, ny, nz = size(u)
    res = Array{T,2}(undef, nv, nz)
    triple_velocity_correlation_vector!(res, ρ₀, u)
    return res
end


# ∂ⱼ(ρ₀⟨uⱼ(uᵢ)²/2⟩)
function triple_velocity_correlation_transport!(
        res::AbstractVector{T}, ρ₀::Real, u::AbstractArray{T,4},
        x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T} 
    ) where {T<:AbstractFloat}
    nv = size(u)[1]; nz = size(res)[1]
    vec = Array{T,2}(undef, nv, nz)
    triple_velocity_correlation_vector!(vec, ρ₀, u)
    res .= ∂1(view(vec, 3, :), z)
    return nothing
end


function triple_velocity_correlation_transport(
        ρ₀::Real, u::AbstractArray{T,4},
        x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T} 
    )::Vector{T} where {T<:AbstractFloat}
    res = Vector{T}(undef, length(z))
    triple_velocity_correlation_transport!(res, ρ₀, u, x, y, z)
    return res
end


# ⟨puⱼ⟩
function pressure_veclocity_corrrelation_vector!(
        res::AbstractArray{T, 2}, p::AbstractArray{T,3}, u::AbstractArray{T,4}
    ) where {T<:AbstractFloat}
    nv, nx, ny, nz = size(u)
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            acc = zero(T) # Mean over i and j
            for j ∈ 1:ny
                for i ∈ 1:nx
                    acc += p[i,j,k]*u[h,i,j,k]
                end
            end
            res[h,k] = acc/(nx*ny)
        end
    end
    return nothing
end


function pressure_veclocity_corrrelation_vector(
        p::AbstractArray{T,3}, u::AbstractArray{T,4}
    ) where {T<:AbstractFloat}
    nv, nx, ny, nz = size(u)
    res = Array{T,2}(2, nv, nz)
    pressure_veclocity_corrrelation_vector!(res, p, u)
    return res
end


# ⟨τᵢⱼuᵢ⟩ = ⟨μ(∇u + ∇uᵀ)ᵢⱼuᵢ⟩ = ⟨μ(∇uᵢⱼ+∇uⱼᵢ)uᵢ⟩
function visc_stress_work_vector!(
        res::AbstractArray{T,2}, 
        ∇u::AbstractArray{T,5}, u::AbstractArray{T,4}, Re::Real
    ) where {T<:AbstractFloat}
    nv, nx, ny, nz = size(∇u[1,:,:,:,:])
    @inbounds @batch for k ∈ 1:nz
        for h ∈ 1:nv
            acc = zero(T) # accumulator for the mean
            for j ∈ 1:ny
                for i ∈ 1:nx
                    acc2 = zero(T) # accumulator for the contraction
                    for g ∈ 1:nv
                        acc2 += (∇u[g,h,i,j,k] + ∇u[h,g,i,j,k])*u[g,i,j,k]
                    end
                    acc += acc2
                end
            end
            res[h,k] = -Re^(-1)*acc/(nx*ny)
        end
    end
    return nothing
end


function visc_stress_work_vector(
        ∇u::AbstractArray{T,5}, u::AbstractArray{T,4}, Re::Real
    )::AbstractArray{T,2} where {T<:AbstractFloat}
    res = Array{T,2}(undef, size(u)[1], size(u)[4])
    visc_stress_work_vector!(res, ∇u, u, Re)
    return res
end


# ∂ⱼ⟨τᵢⱼuᵢ⟩ = ∂₃⟨τᵢ₃uᵢ⟩₁₂
function visc_stress_work_transport!(
        res::AbstractArray{T,1},
        ∇u::AbstractArray{T,5}, u::AbstractArray{T,4}, Re::Real,
        z::AbstractVector{T}
    ) where {T<:AbstractFloat}
    vec = visc_stress_work_vector(∇u, u, Re)
    res .= ∂1(view(vec, 3, :), z)
    return nothing
end


function visc_stress_work_transport(
        ∇u::AbstractArray{T,5}, u::AbstractArray{T,4}, Re::Real,
        z::AbstractVector{T}
    )::Vector{T} where {T<:AbstractFloat}
    res = Vector{T}(undef, size(u)[end])
    visc_stress_work_transport!(res, ∇u, u, Re, z)
    return res
end


# τᵢⱼ
function deviatoric_stress_tensor!(
        res::AbstractArray{T,3}, ∇u::AbstractArray{T,5}, Re::Real
    ) where {T<:AbstractFloat}
    nv, nx, ny, nz = size(∇u[1,:,:,:,:])
    @inbounds @batch for k ∈ 1:nz
        for j ∈ 1:ny
            for i ∈ 1:nx
                for h ∈ 1:nv, g ∈ 1:nv
                    res[g,h,i,j,k] = Re^(-1)*(∇u[g,h,i,j,k] + ∇[h,g,i,j,k])
                end
            end
        end
    end
    return nothing
end


end