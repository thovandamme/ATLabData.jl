# ATLabData

Julia package for loading and handling data from the 
[ATLab](https://github.com/turbulencia/atlab) solver.


## Loading data
Loading is done by _load_, which consists of methods for loading data from visuals.x dns.x, and averages.x output:
```
using ATLData
visuals_data = load("path/to/dir/Buoyancy000000")
raw_data = load("path/to/dir/scal.000000.1")
```

Data of 3 different files can also be loaded into a single vector-like data structure:
```
vector_data = load(
    "path/to/dir/flow.000000.1",
    "path/to/dir/flow.000000.2",
    "path/to/dir/flow.000000.3",
)
```
NetCDF data containing the horizontal averages:
```
var = "Eps"
avg_data = load("/path/to/avgfile", var)
```
The _grid_ file (inigrid.x) has to be present in the same directory.

One can also initialize an empty container from a grid and then use _load!_ 
to write data to that container:
```
grid = loadgrid("/path/to/gridfile")
data = init(grid)
load!("/path/to/file")
```

The grid can also separately be loaded wiht _loadgrid_:
```
grid = loadgrid("/path/to/gridfile")
```

## Data structure
Composite types:
- _Grid_ for data from the gridfile
- _ScalarData_ for data from visuals.x and dns.x output
- _VectorData_ combining three scalar outputs from visuals.x or dns.x - not really
used so far in the package.
- _AveragesData_ for loading the NetCDF output from averages.x

## Handling data
Standard operations are overloaded for _ScalarData_ and _VectorData_:

```
data1 = load("file1")
data2 = load("file2")

data = data1 + data2
data = data1 - data2
data = data1 / data2
data = data1 * data2
```

## Statistics
Basic statistical operations are implemented returning _ScalarData_:
```
data = load("/path/to/file")
meandata = mean(data)
flucsdata = flucs(data)
```