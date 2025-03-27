# maize_architecture

First attempt at a procedurally generated 3D maize canopy architecture
model based on field measurements provided by Matthew Runyun for the
rdla field trials. The L-System used to generate the
geometry is based on the system described in
[Cieslak et al. 2022](https://doi.org/10.1093/insilicoplants/diab039).
Where parameters were missing from the data, values & profiles
from Cieslak et al. 2022 were used.


# Installation

## Conda environment

It is highly recommended that dependencies be installed via the conda 
package manager. Two conda environment YAML files are provided that 
enumerate all of the required dependencies and conda channels. These
files can be used to create
(`conda create -f environment.yml`) or update
(`conda update -n lpy --file environment.yml`)
conda environments with those requirements.

- `environment.yml` dependencies necessary if binaries are available for LPy your system (see note below about Macs w/ Apple Silicon)
- `environment_build_lpy.yml` dependencies necessary if LPy needs to be built from source


## Install on Macs w/ Apple Silicon

Unfortunately, there are not pre-built LPy binaries available for Mac w/
AppleSilicon (ARM64) so it (and one of its dependencies) must be
installed from source (see the "Building from source" section below). The
`environment_build_lpy.yml` file should be used to create & populate 
the conda environment prior to completing the build steps below.


## Dependencies

Installation is complete when you have the following packages installed:

- [LPy](https://lpy.readthedocs.io/en/latest/index.html) for procedural generation of 3D plant geometries
- [yggdrasil](https://github.com/cropsinsilico/yggdrasil) Utilities for manipulating 3D geometry files


## Building from source

### Build PlantGL

With the conda env containing the build dependencies activated...

```
if [ ! -d plantgl ]; then
    git clone git@github.com:openalea/plantgl.git
fi
cd plantgl
if [ ! -d build ]; then
    mkdir build
fi
cd build
cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
      -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBISON_ROOT=${CONDA_PREFIX} \
      -DPython3_ROOT_DIR=${CONDA_PREFIX} \
      -DPython3_FIND_STRATEGY=LOCATION \
      ..
make -j 8
make install
cd ..
pip install .
cd ..
```

This commands are provided by the `build_plantgl.sh` script.


### Build LPy

With the conda env containing the build dependencies activated...

```
export PLANTGL_ROOT="${CONDA_PREFIX}"
if [ ! -d lpy ]; then
    git clone git@github.com:openalea/lpy.git
fi
cd lpy
if [ ! -d build ]; then
    mkdir build
fi
cd build
cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
      -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_ROOT_DIR=${CONDA_PREFIX} \
      -DPython3_FIND_STRATEGY=LOCATION \
      ..
make -j 8
make install
cd ..
pip install .
cd ..
```

This commands are provided by the `build_lpy.sh` script.


# Running

The `maize3D.py` Python script can be used to generate 3D meshes for
single maize plants, plots with multiple rows of maize plants, and
side-by-side plots with different genetic lines. If no arguments are
provided defaults will be assumed (e.g. crop class defaults to "WT"). 
For more information about the allowed parameters, run:

```
python maize3D.py -h
```


## Examples

### Single plant with single crop class

```
python maize3D.py --crop-class=rdla
```


### Single plant with WT & rdla side-by-side

```
python maize3D.py --crop-class=all
```


### 500cm x 500cm plot of uniquely generated WT plants with 100 cm plant spacing between rows & 20 cm plant spacing between plants within rows

```
python maize3D.py --crop-class=WT --canopy=unique --plot-length=500 --plot-width=500 --row-spacing=100 --plant-spacing=20
```


# Repository contents

- `LICENSE` - License for using this model
- `README.md` - This file
- `build_lpy.sh` - Bash script for building LPy from source
- `build_plantgl.sh` - Bash script for building PlantGL from source
- `environment.yml` - Conda environment file containing required dependencies
- `environment_build_lpy.yml` - Conda environment file containing dependencies required for building OpenAlea dependencies (LPy & PlantGL from source)
- `images` - Directory containing images generated from the output meshes
- `input` - Directory containing field measurements
- `maize.lpy` - LPy input file describing the model
- `maize3D.py` - Model interface, I/O tools, & utilities for generating model components
- `meshes` - Directory containing generated 3D meshes
- `param` - Directory containing saved input parameters


# Parameters

| Name            | Description                   | Phenotyping methods |
| ---             | ---                           | ---                 |
| BranchAngle     | Angle between shoot and stem  | Protractor or Image Segmentation  |
| RotationAngle   | Angle between consecutive leaves | Unsure |
| InternodeLength | Distance between nodes           | Ruler or Image segmentation |
| InternodeWidth  | Width of stem between nodes      | Can be estimated from InternodeLength |
| LeafCurve       | Profile of leaf cross-section | Image segmentation |
| LeafBend        | Profile of how leaf curves due to gravity along their length | Image segmentation |
| LeafTwist       | Profile of how leaves twist along their length | Image segmentation |


Most important:
- InternodeLength (or leaf height)
- BranchAngle

Need multiple time points for age dependence, probably 3 or more. Movies along the row (e.g. by rover) could be segmented to get parameters if manual measurements would not be possible. Drone images probably would not help with these parameters, but would be good for validating the canopy coverage predicted by the model.


# Citations

- Mikolaj Cieslak, Nazifa Khan, Pascal Ferraro, Raju Soolanayakanahally, Stephen J Robinson, Isobel Parkin, Ian McQuillan, Przemyslaw Prusinkiewicz, L-system models for image-based phenomics: case studies of maize and canola, in silico Plants, Volume 4, Issue 1, 2022, diab039, https://doi.org/10.1093/insilicoplants/diab039
- F. Boudon, C. Pradal, T. Cokelaer, P. Prusinkiewicz, C. Godin. L-Py: an L-system simulation framework for modeling plant architecture development based on a dynamic language. Frontiers in Plant Science, Frontiers, 2012, 3 (76), doi: 10.3389/fpls.2012.00076


# To do

- Confirm width of mesh leaves (radius vs. diameter)
- Add thickness to leaves when not unfurling
- Describe profiles in README
- More generic way of specifying profiles (e.g. using PlantGL
  editor to generate profile and then saving it to a file that can
  be loaded in the future)
- More realistic age dependency?
- Validate simulated distribution against observations
- Try running hothouse on a small canopy
- Profile to determine if anything can be optimized
- Fix leaf unfurling
- Debug InternodeWidthExp taken from Cieslak et al. 2022 seems to be off by a factor of 10
