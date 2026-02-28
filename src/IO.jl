module IO

using NetCDF
using ..DataStructures

export load, load!, loadgrid, init, header

let 
# ------------------------------------------------------------------------------
#                                   API
# ------------------------------------------------------------------------------

"""
    loadgrid(gridfile) -> Grid
Loads the grid data from the file _gridfile_ into the composite type _grid_.
"""
global loadgrid(gridfile::String)::Grid = _Grid_from_gridfile(gridfile)


"""
    load(file, prec) -> ScalarData
Load the data contained in the path _file_ into the type _Data_.

    load(dir, field, time) -> ScalarData
Search in _dir_ for the file containing _field_ at _time_ and load into the type 
_Data_.

    load(xfile, yfile, zfile) -> VectorData
Load the data contained in the paths _xfile_, _yfile_ and _zfile_ into the 
type _VectorData_.

    load(file, var) -> AveragesData
Load the data contained in the path _file_ into the type _AveragesData_.
_file_ has to be NetCDF file containing the averages from _average.x_.

    load(file, plane, var; vars) -> PlaneData
Load data from a planes _file_ into the container PlaneData. _plane_ is the 
axis index and var the varible which are supposed to be loaded, where var 
needs to the keys in the Dict _vars_. If _vars_ is not given, then it defaults 
to (:u, :v, :w, :b).
"""
global load(
    file::String; prec::Type=Float32
)::ScalarData = _ScalarData_from_file(file, prec)
global load(
    dir::String, field::String, time::Real; component::String=".0"
)::ScalarData = load(_file_for_time(dir, time, field, component))
global load(
    xfile::String, yfile::String, zfile::String; prec::Type=Float32
)::VectorData = _VectorData_from_files(xfile, yfile, zfile, prec)
global load(
    file::String, var::String
)::AveragesData = _AveragesData_from_NetCDF(file, var)
global load(
    file::String, plane::Int, var::Symbol; prec::Type=Float32, 
    vars::Dict=Dict(:u=>1, :v=>2, :w=>3, :b=>4, :p=>5)
) = _PlaneData_from_raw(file, plane, var, prec, vars)


"""
    load!(data, file)
Version of _load_ for preallocated data container. This function does not 
    update the grid attribute!
"""
global load!(
    data::ScalarData{T,I}, file::String
) where {T<:AbstractFloat, I<:Signed} = _ScalarData_from_file!(data, file)
global load!(
    data::VectorData{T,I}, xfile::String, yfile::String, zfile::String
) where {T<:AbstractFloat, I<:Signed} = _VectorData_from_file!(data, xfile, yfile, zfile)
global load!(
    data::PlaneData{T,I}, file::String, plane::Int, var::Symbol; 
    vars::Dict=Dict(:u=>1, :v=>2, :w=>3, :b=>4, :p=>5)
) where {T<:AbstractFloat, I<:Signed} = _PlaneData_from_raw!(data, file, plane, var, vars)


"""
    init(grid; dims=1) -> AbstractData
Initialize an empty data container. dims has to be an integer in of value 
1 or 3. 1 correpsonds to _ScalarData_ while 3 returns _VectorData_. 
Default for _dims_ is 1. _grid_ has to be of type _Grid_.
"""
global function init(
        grid::Grid{T,I};
        dims::Int=1
    )::AbstractData where {T<:AbstractFloat, I<:Signed}
    if dims==1
        return ScalarData(
            name = "empty container",
            grid = grid,
            time = convert(T, 0.0),
            field = Array{T, 3}(undef, grid.nx, grid.ny, grid.nz)
        )
    elseif dims==3
        return VectorData(
            name = "empty container",
            grid = grid,
            time = convert(T, 0.0),
            xfield = Array{T, 3}(undef, grid.nx, grid.ny, grid.nz),
            yfield = Array{T, 3}(undef, grid.nx, grid.ny, grid.nz),
            zfield = Array{T, 3}(undef, grid.nx, grid.ny, grid.nz),
        )
    end
end


global function init(gridfile::String; dims::Int=1, T::Type=Float64)::AbstractData
    return init(convert(T, loadgrid(gridfile)), dims=dims)
end


# ------------------------------------------------------------------------------
#                                   Grid
# ------------------------------------------------------------------------------
""" 
    Load the file _grid_ into composite type _Grid_
"""
function _Grid_from_gridfile(gridfile::String)::Grid
    "f90_record_markers from the amount of write commands in storing routine "
    io = open(gridfile, "r")
    marker1 = read(io, Int32)
    nx = read(io, Int32)
    ny = read(io, Int32)
    nz = read(io, Int32)
    marker2 = read(io, Int32)
    marker3 = read(io, Int32)
    scalex = read(io, Float64)
    scaley = read(io, Float64)
    scalez = read(io, Float64)
    marker4 = read(io, Int32)
    marker5 = read(io, Int32)
    x = Vector{Float64}(undef, nx)
    for i ∈ 1:nx
        x[i] = read(io, Float64)
    end
    marker7 = read(io, Int32)
    marker8 = read(io, Int32)
    y = Vector{Float64}(undef, ny)
    for i ∈ 1:ny
        y[i] = read(io, Float64)
    end
    marker9 = read(io, Int32)
    marker10 = read(io, Int32)
    z = Vector{Float64}(undef, nz)
    for i ∈ 1:nz
        z[i] = read(io, Float64)
    end
    return Grid(nx, ny, nz, scalex, scaley, scalez, x, y, z)
end


function _Grid_from_file(dir::String)::Grid
    gridfile = joinpath(dir, "grid")
    return _Grid_from_gridfile(gridfile)
end


# ------------------------------------------------------------------------------
#                               ScalarData
# ------------------------------------------------------------------------------
"""
    Loading data that is stored in a single file.
"""
function _ScalarData_from_file(fieldfile::String, type::Type)::ScalarData
    verbose("ScalarData", fieldfile)
    filename = split(fieldfile, "/")[end]
    if startswith(filename, "flow.") || startswith(filename, "scal.")
        return _ScalarData_from_raw(fieldfile, type)
    else
        return _ScalarData_from_visuals(fieldfile)
    end
end


function _ScalarData_from_raw(
        fieldfile::String, T::Type
    )::ScalarData{T, Int32}
    grid = convert(T, _Grid_from_file(dirname(fieldfile)))
    buffer, t = _Array_from_rawfile(grid, fieldfile)
    return ScalarData(
        name = splitpath(fieldfile)[end],
        grid = grid,
        time = t,
        field = buffer
    )
end


# function _ScalarData_from_raw(
#         fieldfile::String, T::Type
#     )::ScalarData{T, Int32}
#     io = open(fieldfile, "r")
#     grid = convert(T, _Grid_from_file(dirname(fieldfile)))
#     h = convert(T, _fieldheader(io))
#     # (h.nx!=grid.nx || h.ny!=grid.ny || h.nz!=grid.nz) && error("Grid size mismatch!")
#     buffer = _Array_from_rawfile_(io, h)
#     close(io)
#     return ScalarData(
#         name = basename(fieldfile),
#         grid = grid,
#         header = h,
#         field = buffer
#     )
# end


function _ScalarData_from_visuals(fieldfile::String)::ScalarData{Float32, Int32}
    grid = convert(Float32, _Grid_from_file(dirname(fieldfile)))
    return ScalarData(
        name = splitpath(fieldfile)[end],
        grid = grid,
        time = _time_from_file(fieldfile),
        field = _Array_from_file(grid, fieldfile)
    )
end


# TODO add prec argument to load scal and flow with Float32
function _ScalarData_from_file!(data::ScalarData, fieldfile::String)
    filename = split(fieldfile, "/")[end]
    verbose("ScalarData", fieldfile)
    data.name = filename
    if startswith(filename, "flow.") || startswith(filename, "scal.")
        try
            data.field, data.time = _Array_from_rawfile(data.grid, fieldfile)
        catch e
            println("Got $e")
            println(
                "Did you initialize the container with the same floating 
                point precission as in the file?"
            )
        end
    else
        data.time = _time_from_file(fieldfile)
        data.field .= _Array_from_file(data.grid, fieldfile)
    end
    return nothing
end


# ------------------------------------------------------------------------------
#                               VectorData
# ------------------------------------------------------------------------------
"""
    Loading data from three files. No check for physical consistency.
"""
function _VectorData_from_files(
        xfieldfile::String,
        yfieldfile::String,
        zfieldfile::String,
        prec::Type
    )::VectorData
    verbose("VectorData", xfieldfile, yfieldfile, zfieldfile)
    filename = split(xfieldfile, "/")[end]
    if startswith(filename, "flow.") || startswith(filename, "scal.")
        return _VectorData_from_raw(xfieldfile, yfieldfile, zfieldfile, prec)
    else
        return _VectorData_from_visuals(xfieldfile, yfieldfile, zfieldfile)
    end
end


function _VectorData_from_raw(
        xfieldfile::String,
        yfieldfile::String,
        zfieldfile::String,
        T::Type
    )::VectorData{T, Int32}
    grid = convert(T,_Grid_from_file(dirname(xfieldfile)))
    field = Array{T}(undef, 4, grid.nx, grid.ny, grid.nz)
    field[1,:,:,:], t = _Array_from_rawfile(grid, xfieldfile)
    field[2,:,:,:] .= _Array_from_rawfile(grid, yfieldfile)[1]
    field[3,:,:,:] .= _Array_from_rawfile(grid, zfieldfile)[1]
    return VectorData(
        name = string(splitpath(xfieldfile)[end][1:end-2]),
        grid = grid,
        time = t,
        field = field
    )
end


function _VectorData_from_visuals(
        xfieldfile::String,
        yfieldfile::String,
        zfieldfile::String
    )
    grid = convert(Float32, _Grid_from_file(dirname(xfieldfile)))
    field = Array{T}(undef, 4, grid.nx, grid.ny, grid.nz)
    field[1,:,:,:] .= _Array_from_file(grid, xfieldfile)
    field[2,:,:,:] .= _Array_from_file(grid, yfieldfile)
    field[3,:,:,:] .= _Array_from_file(grid, zfieldfile)
    return VectorData(
        name = string(splitpath(xfieldfile)[end][1:end-2]),
        grid = grid,
        time = _time_from_file(xfieldfile),
        field = field
    )
end


function _VectorData_from_files!(
        data::VectorData, 
        xfieldfile::String,
        yfieldfile::String,
        zfieldfile::String
    )
    filename = split(xfieldfile, "/")[end]
    verbose("VectorData", xfieldfile, yfieldfile, zfieldfile)
    data.name = string(splitpath(xfieldfile)[end][1:end-2])
    if startswith(filename, "flow.") || startswith(filename, "scal.")
        try 
            data.field[1,:,:,:], data.time = _Array_from_rawfile(data.grid, xfieldfile)
            data.field[2,:,:,:] .= _Array_from_rawfile(data.grid, yfieldfile)[1]
            data.field[3,:,:,:] .= _Array_from_rawfile(data.grid, zfieldfile)[1]
        catch e
            prinlnt("$e")
            println(
                "Did you initialize the container with the same floating 
                point precission as in the file?"
            )
        end
    else
        data.time = _time_from_file(fieldfile)
        data.field[1,:,:,:] .= _Array_from_file(data.grid, xfieldfile)
        data.field[2,:,:,:] .= _Array_from_file(data.grid, yfieldfile)
        data.field[3,:,:,:] .= _Array_from_file(data.grid, zfieldfile)
    end
    return nothing
end


# ------------------------------------------------------------------------------
#                         Array from binary
# ------------------------------------------------------------------------------
"""
    Load array from a binary file according to the information in _grid_.
"""
function _Array_from_file(
        grid::Grid{T,I}, fieldfile::String
    )::Array{T,3} where {T<:AbstractFloat, I<:Signed}
    buffer = Vector{T}(undef, grid.nx*grid.ny*grid.nz)
    read!(fieldfile, buffer)
    return reshape(buffer, (grid.nx, grid.ny, grid.nz))
end
 

function _Array_from_rawfile(
        grid::Grid{T,I}, fieldfile::String
    )::Tuple{Array{T, 3}, T} where {T<:AbstractFloat, I<:Signed}
    io = open(fieldfile, "r")
    """ 
        Header contains: 
        headersize/offset, nx, ny, nz, nt, time, visc, froude, schmidt
        unformatted with no record markers
    """
    headersize = read(io, Int32) # First entry is the header length in bytes
    seek(io, 5*sizeof(headersize))
    time = read(io, Float64)
    seek(io, headersize) # Jump to first last entry belonging to the header
    buffer = Vector{T}(undef, grid.nx*grid.ny*grid.nz)
    read!(io, buffer)
    close(io)
    return reshape(buffer, (grid.nx, grid.ny, grid.nz)), time
end


function _Array_from_rawfile_(
        io::IOStream, h::FieldHeader{T,I}
    )::Array{T, 3} where {T<:AbstractFloat, I<:Signed}
    buffer = Vector{T}(undef, h.nx*h.ny*h.nz)
    seek(io, h.headersize)
    read!(io, buffer)
    close(io)
    return reshape(buffer, (h.nx, h.ny, h.nz))
end


# ------------------------------------------------------------------------------
#                            AveragesData
# ------------------------------------------------------------------------------
function _AveragesData_from_NetCDF(file::String, var::String)::AveragesData
    return AveragesData(
        name = var,
        time = ncread(file, "t"),
        z = ncread(file, "z"),
        field = ncread(file, var)
    )
end


# ------------------------------------------------------------------------------
#                               PlaneData
# ------------------------------------------------------------------------------
function _PlaneData_from_raw(
        file::String, plane::Int, var::Symbol, prec::Type, vars::Dict
    )::PlaneData{prec,Int32}
    # Inspired by planes2planes.py from atlab
    """
        Header: headersize::Int32, iteration::Int32, time::Float32, planeindices::Int32
        data: variable -> planes (data is ordered by variables and then by plane indices)
        variables: TODO In which order are they stored?
        The dictinoary vars containes the indices for the varibales, that can be 
            given by var.
    """
    !haskey(vars, var) && error("var has to be a key of the dictionary vars.")
    grid = convert(prec, loadgrid(joinpath(dirname(file), "grid")))
    planesize = 0; n1 = 0; n2 = 0
    filename = basename(file)
    if split(filename, ".")[1]=="planesI"
        planesize = grid.ny*grid.nz
        grid.nx = 1
        val = grid.x[plane]
        grid.x = ones(prec, 1)*val
        n1 = grid.ny
        n2 = grid.nz
    elseif split(filename, ".")[1]=="planesJ"
        planesize = grid.nx*grid.nz
        grid.ny = 1
        val = grid.y[plane]
        grid.y = ones(prec, 1)*val
        n1 = grid.nx
        n2 = grid.nz
    elseif split(filename, ".")[1]=="planesK"
        planesize = grid.nx*grid.ny
        grid.nz = 1
        val = grid.z[plane]
        grid.z = ones(prec, 1)*val
        n1 = grid.nx
        n2 = grid.ny
    else
        error("Do not know how to handle this planes-file.")
    end
    io = open(file, "r")
    h = _planesheader(io)
    !(plane ∈ h.planes) && error(
        "Plane with index $plane is not in $file. Available planes are $(h.planes)"
    )
    # Number of variables
    fieldbytes = filesize(io) - h.headersize
    planebytes = planesize*sizeof(prec)
    varbytes = length(h.planes)*planebytes
    nvars = fieldbytes÷varbytes
    (nvars != length(vars)) && (
        @warn begin
            "The number of variables in vars is not the number of variables in $file.\n"
            "Number of variables in $file: \n $nvars"
        end
    )
    # Find position of plane in file and read into buffer
    iplane = argmin(abs.(h.planes .- plane))
    seek(
        io, 
        h.headersize + ((vars[var]-1)*length(h.planes)+(iplane-1))*planebytes
    )
    buffer = Vector{prec}(undef, planesize)
    read!(io, buffer)
    close(io)
    return PlaneData(
        name = "$plane-$var-$(basename(file))",
        header = convert(prec, h),
        grid = grid,
        field = reshape(buffer, (n1, n2))
    )
end


function _PlaneData_from_raw!(
        data::PlaneData{T,I}, file::String, plane::Int, var::Symbol, vars::Dict
    ) where {T<:AbstractFloat, I<:Signed}
    !haskey(vars, var) && error("var has to be a key of the dictionary vars.")
    filename = basename(file)
    if split(filename, ".")[1]=="planesI"
        n1 = data.grid.ny
        n2 = data.grid.nz
    elseif split(filename, ".")[1]=="planesJ"
        n1 = data.grid.nx
        n2 = data.grid.nz
    elseif split(filename, ".")[1]=="planesK"
        n1 = data.grid.nx
        n2 = data.grid.ny
    else
        error("Do not know how to handle this planes-file.")
    end
    io = open(file, "r")
    h = _planesheader(io)
    !(plane ∈ h.planes) && error(
        "Plane with index $plane is not in $file. Available planes are $(h.planes)"
    )
    # Number of variables
    planesize = length(data.field)
    fieldbytes = filesize(io) - h.headersize
    planebytes = planesize*sizeof(T)
    varbytes = length(h.planes)*planebytes
    nvars = fieldbytes÷varbytes
    (nvars != length(vars)) && (
        @warn begin
            "The number of variables in vars is not the number of variables in $file.\n"
            "Number of variables in $file: \n $nvars"
        end
    )
    # Find position of plane in file and read into buffer
    iplane = argmin(abs.(h.planes .- plane))
    seek(
        io, 
        h.headersize + ((vars[var]-1)*length(h.planes)+(iplane-1))*planebytes
    )
    buffer = Vector{T}(undef, planesize)
    read!(io, buffer)
    close(io)
    data.field .= reshape(buffer, (n1, n2))
    return nothing
end


# ------------------------------------------------------------------------------
#                             Header loading
# ------------------------------------------------------------------------------
"""
    header(file)
Returns the header of _file_.
"""
global function header(file::String)
    filename = basename(file)
    io = open(file, "r")
    if startswith(filename, "scal.") || startswith(filename, "flow.")
        h = _fieldheader(io)
        close(io)
        return h
    elseif startswith(filename, "planes")
        h = _planesheader(io)
        close(io)
        return h
    else
        error("Do not know how to get the header of this file.")
    end
end


function _fieldheader(io::IOStream)::FieldHeader{Float64,Int32}
    """ 
        Header contains: 
        headersize/offset, nx, ny, nz, nt, time, visc, froude
        unformatted with no record markers
    """
    headersize = read(io, Int32)
    nx = read(io, Int32)
    ny = read(io, Int32)
    nz = read(io, Int32)
    iteration = read(io, Int32)
    time = read(io, Float64)
    bytesofparams = headersize-sizeof(headersize)
    bytesofparams = bytesofparams-sizeof(nx)- sizeof(ny)-sizeof(nz)
    bytesofparams = bytesofparams-sizeof(iteration)- sizeof(time)
    params = reinterpret(Float64, read(io, bytesofparams))
    return FieldHeader{Float64,Int32}(
        headersize, nx, ny, nz, iteration, time, params
    )
end


function _planesheader(io::IOStream)::PlanesHeader{Float64,Int32}
    headersize = read(io, Int32)
    iteration = read(io, Int32)
    time = read(io, Float64)
    setofplanes = reinterpret(Int32, read(io, (headersize-4-4-8)))
    return PlanesHeader{Float64,Int32}(
        headersize, iteration, time, setofplanes
    )
end


# ------------------------------------------------------------------------------
#                          Helping functions
# ------------------------------------------------------------------------------
function _timestep_from_filename(filename::String)::Int
    namestring = split(filename, "/")[end]
    namestring = split(namestring, ".")[1]
    stepstring = namestring[end-5:end]
    return parse(Int, stepstring)
end


function _time_from_timestep(timestep::Int, dir::String)::AbstractFloat
    avgfile = "avg"*string(timestep)*".nc"
    avgfile = joinpath(dir, avgfile)
    return ncread(avgfile, "t")[1]
end


function _time_from_file(file::String)::Float32
    tstep = _timestep_from_filename(file)
    t = _time_from_timestep(tstep, dirname(file))
    return t
end


function _file_for_time(
        dir::String, 
        time::Real,
        field::String,
        component::String=".0"
    )::String
    Δt = time
    f = ""
    maxΔt_relative = 0.1
    maxΔt = maxΔt_relative*time
    println("Searching for t = ", time)
    for filename in readdir(dir)
        correct_field = false
        if component ∈ (".1", ".2", ".3", ".4", ".5", ".6")
            if startswith(filename, field) && endswith(filename, component)
                correct_field = true
            end
        elseif component == ".0"
            if startswith(filename, field)
                correct_field = true
            end
        else
            error("Loading a field with this _component_ is not implemented.")
        end
        if correct_field
            file = joinpath(dir, filename)
            t = _time_from_file(file)
            printstyled("   ", filename, "    ", t, "\n", color=:cyan)
            if abs(time - t) < Δt
                f = file
                Δt = abs(time - t)
            end
        end
    end
    if Δt > maxΔt
        @warn "The time loaded exceeds the maximal accepted relative error of $(maxΔt_relative)."
    end
    return f
end


function verbose(dtype::String, files...; save::Bool=false)
    if !save
        text = "Loading " * dtype * " ..."
    else
        text = "Saving " * dtype * " ..."
    end
    println(text)
    for file in files
        printstyled("   "*file*"\n", color=:cyan)
    end
end


function done()
    printstyled("   Done \n", color=:green)
end


end # end of let encapsulation


end