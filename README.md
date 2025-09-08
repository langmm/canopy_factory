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
(`conda env create --file environment.yml`) or update
(`conda env update -n canopy --file environment.yml`)
conda environments with those requirements.

- `environment.yml` dependencies necessary if binaries are available for LPy your system (see note below about Macs w/ Apple Silicon)
- `environment_build_lpy.yml` dependencies necessary if LPy needs to be built from source


## Install on Macs w/ Apple Silicon

Unfortunately, there are not pre-built LPy binaries available for Mac w/
AppleSilicon (ARM64) so it (and one of its dependencies) must be
installed from source (see the "Installing from source" section below). The
`environment_build_lpy.yml` file should be used to create & populate 
the conda environment prior to completing the build steps below.


## Dependencies

Installation is complete when you have the following packages installed:

- [LPy](https://lpy.readthedocs.io/en/latest/index.html) for procedural generation of 3D plant geometries
- [yggdrasil](https://github.com/cropsinsilico/yggdrasil) Utilities for manipulating 3D geometry files


## Installing from source

### Install PlantGL

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

These commands are provided by the `build_plantgl.sh` script.


### Install LPy

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

These commands are provided by the `build_lpy.sh` script.


## Installing ray tracer

If you would like to run a ray tracer on the generated 3D geometries,
`maize3d.py` has some options for doing so using [pyembree](https://github.com/scopatz/pyembree) and [hothouse](https://github.com/cropsinsilico/hothouse). Both of these libraries will need to be installed from source as they are not under active development and had to be updated to use embree 4. Before running the command below, the conda environment should be updated with their dependencies that can be found in `environment_raytrace.yml`:

```
conda update -n canopy --file environment_raytrace.yml
```

If you are using Python >= 3.12 and you get an error along the lines of `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?` during the installation process, you will need to
manually upgrade pip and setuptools via:

```
python -m ensurepip --upgrade
python -m pip install --upgrade setuptools
```

### Installing pyembree

```
git clone git@github.com:langmm/pyembree.git
cd pyembree
pip install . --no-build-isolation
```

These commands are provided by the `build_pyembree.sh` script.


### Installing hothouse

```
git clone --branch maize git@github.com:langmm/hothouse.git
cd hothouse
pip install . --no-build-isolation
```

These commands are provided by the `build_hothouse.sh` script.


# Running

The `canopy_factor` Python CLI can be used to generate 3D meshes for
single maize plants, plots with multiple rows of maize plants, and
side-by-side plots with different genetic lines. If no arguments are
provided defaults will be assumed (e.g. id defaults to "WT"). 
For more information about the allowed parameters, run:

```
python -m canopy_factory -h
```


## Examples

### Single plant with single id

```
python -m canopy_factory --id=rdla
```


### Single plant with WT & rdla side-by-side

```
python -m canopy_factory --id=all
```


### 500cm x 500cm plot of uniquely generated WT plants with 100 cm plant spacing between rows & 20 cm plant spacing between plants within rows

```
python -m canopy_factory --id=WT --canopy=unique --plot-length=500 --plot-width=500 --row-spacing=100 --plant-spacing=20
```


# Repository contents

- `LICENSE` - License for using this model
- `README.md` - This file
- `build_embree.sh` - Bash script for building embree from source
- `build_hothouse.sh` - Bash script for building hothouse from source
- `build_lpy.sh` - Bash script for building LPy from source
- `build_plantgl.sh` - Bash script for building PlantGL from source
- `build_pyembree.sh` - Bash script for building pyembree from source
- `canopy_factory` - Python package containing model interface, I/O tools, & utilities for generating model components
- `environment.yml` - Conda environment file containing required dependencies
- `environment_build_lpy.yml` - Conda environment file containing dependencies required for building OpenAlea dependencies (LPy & PlantGL from source)
- `environment_raytrace.yml` - Conda environment file containing required dependencies for building embree, pyembree & hothouse
- `tests` - Package tests
- `yamls` - YAML specification files for yggdrasil

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
- Describe profiles in README
- More generic way of specifying profiles (e.g. using PlantGL
  editor to generate profile and then saving it to a file that can
  be loaded in the future)
- More realistic age dependency for internode length
- Validate simulated distribution against observations
- Profile to determine if anything can be optimized
- Fix leaf unfurling
- Debug InternodeWidthExp taken from Cieslak et al. 2022 seems to be off by a factor of 10
- Connect to yggdrasil
- Add to model repo
- Add mass dependency for limiting growth
- integrate with BioCro
- options for labeling crop class & plant id
- allow for other crops
- allow for inputs to be specified for input parameters and data
- gravitropism
- add tests
- publish to PyPI
- publish to conda_forge
- build docs