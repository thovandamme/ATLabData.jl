"""
Module to load and process data from ATLab in Julia.
"""
module ATLabData

using Reexport

include("DataStructures.jl")
@reexport using .DataStructures

include("IO.jl")
@reexport using .IO

include("Basics.jl")
@reexport using .Basics

# include("Analysis.jl")
# @reexport using .Analysis

include("FDM.jl")
using .FDM

include("Calculus.jl")
@reexport using .Calculus

include("Statistics.jl")
@reexport using .Statistics

include("Tools.jl")
@reexport using .Tools

include("Physics.jl")
@reexport using .Physics


function __init__()
    # 
end


end