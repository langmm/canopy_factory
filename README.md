# canopy_factory

First attempt at a procedurally generated 3D canopy architecture
model including a maize model based on field measurements provided by
Matthew Runyun for the rdla field trials. The L-System used to generate
the geometry is based on the system described in
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
- `environment_build_lpy.yml` dependencies necessary if LPy needs to be built from source for your OS

## Dependencies installed from source or outside PyPI

There are several packages that may need to be install via source or from a package manager other than python's pip/conda. Where possible, scripts are provided to aid in completing these steps if they are necessary.

- [hothouse](https://github.com/cropsinsilico/hothouse) for using the embree ray tracer to calculate the light intercepted by generated geometries. Must be built from source until a release is available on PyPI (ETA late spring or summer 2026).
- [OpenAlea PlantGL](https://github.com/openalea/plantgl) for creating virtual plant geometry meshes (required by LPy). Must be built from source if conda not used.
- [OpenAlea LPy](https://lpy.readthedocs.io/en/latest/index.html) for procedural generation of 3D plant geometries. Must be built from source if conda not used.
.
- [ffmpeg](https://github.com/ffmpeg/ffmpeg) for creating animations. Must be installed via a package manager for your OS (e.g. homebrew, apt, vcpkg) if conda is not used.


### Installing hothouse

```
git clone --branch maize git@github.com:langmm/hothouse.git
cd hothouse
pip install . --no-build-isolation
```

These commands are provided by the `build_hothouse.sh` script.


### Install PlantGL

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


## Install canopy_factory

*After* installing the non-PyPI dependencies via one of the two methods below, `canopy_factory` can be installed in development mode via:

```
python -m pip install -e '.[dev]'
```

## Possible installation errors

If you are using Python >= 3.12 and you get an error along the lines of `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?` during the installation process, you will need to
manually upgrade pip and setuptools via:

```
python -m ensurepip --upgrade
python -m pip install --upgrade setuptools
```

# Running

The `canopy_factory` Python CLI can be used to generate and/or analyze 3D meshes for
single plants, plots with multiple rows of plants, or
side-by-side plots with different genetic lines. If no arguments are
provided defaults will be assumed (e.g. id defaults to "B73_WT" for maize). 
For more information about the allowed parameters, run:

```
python -m canopy_factory -h
```


## Examples

### Single maize plant using data collected in 2024 for B73 rdla

```
python -m canopy_factory generate maize --id=B73_rdla
```


### Single maize plant using data collected in 2024 for B73 WT & rdla side-by-side

```
python -m canopy_factory generate maize --id=all_combined
```


### 500cm x 500cm plot of uniquely generated B73 WT maize plants with 100 cm plant spacing between rows & 20 cm plant spacing between plants within rows

```
python -m canopy_factory generate maize --id=B73_WT --canopy=unique --plot-length=500 --plot-width=500 --row-spacing=100 --plant-spacing=20
```

### Plot light totals for all IDs in the maize data file for 2025 using periodic boundary conditions

```
python -m canopy_factory totals maize --canopy=unique --periodic-canopy --output-totals-plot --id=all --data-year=2025
```


### Plot the layout for a field with periodic boundary conditions

```
python -m canopy_factory layout --periodic-canopy
```

### Movie over one day for plot of raytraced unique B73 WT maize plants generated using data from 2025

```
python -m canopy_factory animate maize --canopy=unique --id=B73_WT --data-year=2025
```

### Movie over growing season for plot of raytraced unique B73 WT maize plants generated using data from 2025

```
python -m canopy_factory animate maize --canopy=unique --id=B73_WT --data-year=2025 --start-date=planting --stop-date=maturity
```

### Find the row spacing the makes the light intercepted by a plot of B73_rdla match the light intercepted by plot of B73_WT with a row spacing of 76.2

```
python -m canopy_factory match_query maize --id=B73_rdla --vary=row_spacing
```

# Documentation

Additional documentation on the Python APIs can be found [here](https://langmm.github.io/canopy_factory/).

# Repository contents

- `LICENSE` - License for using this model
- `README.md` - This file
- `pyproject.toml` - Python package configuration file
- `build_embree.sh` - Bash script for building embree from source
- `build_hothouse.sh` - Bash script for building hothouse from source
- `build_lpy.sh` - Bash script for building LPy from source
- `build_plantgl.sh` - Bash script for building PlantGL from source
- `build_pyembree.sh` - Bash script for building pyembree/embreex from source
- `canopy_factory` - Python package containing model interface, I/O tools, & utilities for generating model components
- `environment.yml` - Conda environment file containing required dependencies
- `environment_build_lpy.yml` - Conda environment file containing dependencies required for building OpenAlea dependencies (LPy & PlantGL from source)
- `environment_raytrace.yml` - Conda environment file containing required dependencies for building embree, pyembree/embreex & hothouse
- `tests` - Package tests
- `yamls` - YAML specification files for yggdrasil
- `docs` - Package documentation

# Funding

This work is supported by funding from the CROPPS NSF STC (Grant No. DBI-2019674).

# Citations

- Mikolaj Cieslak, Nazifa Khan, Pascal Ferraro, Raju Soolanayakanahally, Stephen J Robinson, Isobel Parkin, Ian McQuillan, Przemyslaw Prusinkiewicz, L-system models for image-based phenomics: case studies of maize and canola, in silico Plants, Volume 4, Issue 1, 2022, diab039, https://doi.org/10.1093/insilicoplants/diab039
- F. Boudon, C. Pradal, T. Cokelaer, P. Prusinkiewicz, C. Godin. L-Py: an L-system simulation framework for modeling plant architecture development based on a dynamic language. Frontiers in Plant Science, Frontiers, 2012, 3 (76), doi: 10.3389/fpls.2012.00076


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

# To do

- Confirm width of mesh leaves (radius vs. diameter)
- Describe profiles in README
- More generic way of specifying profiles (e.g. using PlantGL
  editor to generate profile and then saving it to a file that can
  be loaded in the future)
- More realistic age dependency for internode length
- Validate simulated distribution against observations
- Profile to determine if anything can be optimized
- Parallelize ray tracing
- Fix leaf unfurling
- Debug InternodeWidthExp taken from Cieslak et al. 2022 seems to be off by a factor of 10
- Connect to yggdrasil
- Add to model repo
- Add mass dependency for limiting growth
- integrate with BioCro
- gravitropism
- phototropism
- publish to PyPI
- publish to conda_forge
- build docs
- Allow leaf reflectance/transmittance to be provided as spectra
- More realistic age dependency for color change & senesence
- Only include green leaves in LAI calculation
