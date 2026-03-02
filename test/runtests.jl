using ATLabData
using Test

@testset "ATLabData.jl" begin
    @testset "Grid" begin
        grid = loadgrid("grid")
        printstyled("Grid tests passed ✓ \n", bold=true, color=:green)
    end
    @testset "ScalarData" begin
        file = "scal.10.1"
        data = load(file, prec=Float64)
        data = convert(Float32, data) 
        data = crop(data)
        data2 = gradient(data)
        gradient!(data2, data)
        data*data^2 - data + 2*data/data
        sqrt(abs(data))
        printstyled("ScalarData tests passed ✓ \n", bold=true, color=:green)
    end
    @testset "VectorData" begin
        data = load("flow.10.1", "flow.10.2", "flow.10.3", prec=Float64)
        data = convert(Float32, data)
        data2 = curl(data)
        curl!(data2, data)
        # data2 = divergence(data)
        # divergence!(data2, data)
        data - data + 2*data/7.0
        abs(data)
        printstyled("VectorData tests passed ✓ \n", bold=true, color=:green)
    end
end