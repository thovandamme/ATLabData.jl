module Calculus

# Overloads ArrayCalculus.jl for the composite types in DataStructures.jl

using ..DataStructures, ..Reexport, ..ArrayCalculus
import ArrayCalculus: 
    ∂x, ∂x!, ∂y, ∂y!, ∂z, ∂z!,∂1, ∂1!, ∂2, ∂2!, ∂3, ∂3!, gradient, curl

export ∂1, ∂2, ∂3, ∂x, ∂y, ∂z, gradient, curl

∂1(data::ScalarData) = ∂1(data.field, data.grid.x)
∂2(data::ScalarData) = ∂2(data.field, data.grid.y)
∂3(data::ScalarData) = ∂3(data.field, data.grid.z)

∂x(data::ScalarData) = ∂1(data)
∂y(data::ScalarData) = ∂2(data)
∂z(data::ScalarData) = ∂3(data)

gradient(data::ScalarData) = gradient(
    data.field, data.grid.x, data.grid.y, data.grid.z
)
curl(data::VectorData) = curl(data.field, data.grid.x, data.grid.y, data.grid.z)



end # module Calculus