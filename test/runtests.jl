using ATLabData
using Test

@testset "ATLabData.jl" begin
    grid = loadgrid("/home/thomas/simulations/tmp/grid")
    data = convert(Float32, init(grid))
    load!(data, "/home/thomas/simulations/tmp/scal.10000.1")
    @time buffer = ATLabData.Basics._crop(data, zmin=10, zmax=60)
    @time buffer = ATLabData.Basics._crop(data, zmin=10, zmax=60)
    @time buffer = ATLabData.Basics._crop(data, zmin=10, zmax=60)
    println("-----")
    @time buffer2 = ATLabData.Basics.crop(data, zmin=10, zmax=60)
    @time buffer2 = ATLabData.Basics.crop(data, zmin=10, zmax=60)
    @time buffer2 = ATLabData.Basics.crop(data, zmin=10, zmax=60)
end
