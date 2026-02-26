using ATLabData
using Test

@testset "ATLabData.jl" begin
    test_Grid()
    test_ScalarData()
    test_VectorData()
    test_PlaneData()
end

function test_Grid()
    # TODO
end

function test_ScalarData()
    println("Running tests for ScalarData ...")
    file = "files/scal.0.1"
    data = load(file)
    data = convert(Float32, data)
    data = crop(data)
    gradient(data)
    data*data^2 - data + 2*data/data
    abs(data)
    return nothing
end

function test_VectorData()
    # TODO
end

function test_PlaneData()
    # TODO
end