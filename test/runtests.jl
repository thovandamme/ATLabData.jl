using ATLabData
using Test


@testset "Basics.jl" begin
    grid = loadgrid("grid")
    # ScalaData
    file = "scal.10.1"
    data = load(file, prec=Float64)
    data = init(grid)
    load!(data, file)
    data = convert(Float32, data) 
    data = crop(data)
    data*data^2 - data + 2*data/data
    sqrt(abs(data))

    # VectorData
    data = load("flow.10.1", "flow.10.2", "flow.10.3", prec=Float64)
    data = convert(Float32, data)
    data - data + 2*data/7.0
    abs(data)

    printstyled("✓ IO.jl and Basics.jl passed \n", bold=true, color=:green)
end


@testset "Statistics.jl" begin
    file = "scal.10.1"
    data = load(file, prec=Float64, verbose=false)
    data = convert(Float32, data)
    # TODO
    printstyled("✓ Statistics.jl tests passed \n", bold=true, color=:green)
end


@testset "Calculus.jl" begin
    # ScalaData
    file = "scal.10.1"
    data = load(file, prec=Float64, verbose=false)
    @time ∂x(data)
    @time ∂z(data)
    @time data2 = gradient(data)
    @time gradient!(data2, data)

    # 1D arrays
    @time ∂x(data.field[:,1,107], data.grid.x)
    @time ∂z(data.field[3,1,:], data.grid.z)
    @time ∂x2(data.field[:,1,107], data.grid.x)
    @time ∂z2(data.field[3,1,:], data.grid.z)

    # VectorData
    data = load("flow.10.1", "flow.10.2", "flow.10.3", prec=Float64)
    data = convert(Float32, data)
    @time data2 = curl(data)
    @time curl!(data2, data)
    data2 = divergence(data)
    divergence!(data2, data)

    printstyled("✓ Calculus.jl tests passed \n", bold=true, color=:green)
end


@testset "Physics.jl" begin
    # TODO
end