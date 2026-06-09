# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.2.0](https://github.com/langmm/canopy_factory/releases/tag/v0.2.0) - 2026-06-09

<small>[Compare with v0.1.0](https://github.com/langmm/canopy_factory/compare/v0.1.0...v0.2.0)</small>

Initial release for PyPI.

### Summary of major features

- Coverage & LAI calculations for simulated canopies
- Virtual canopies for more efficient ray tracigin
- Package maintenance including GHA workflows for publishing docs & Python wheels to PyPI
- Fix to rounding errors in image resolution calculation
- Updates for use with initial release of hothouse
- Updates for use of latest openalea.lpy

### Added

- Add test for non-periodic virtual canopy ([2e56d40](https://github.com/langmm/canopy_factory/commit/2e56d40bd26858e5a060e3970d90c9d2eae2461a) by Meagan Lang).
- Add additional scene & raytracer info to stats ([5aa9b70](https://github.com/langmm/canopy_factory/commit/5aa9b70622c723d5e366401d24eb660ad8661852) by Meagan Lang).
- Add virtual/periodic shifts to raytrace stats ([91a964b](https://github.com/langmm/canopy_factory/commit/91a964b575a7722a5793c11d9366ee45dccff27c) by Meagan Lang).
- Added missing help info for tasks Use hothouse.sun_calc.stable_tan for raytracer camera calculations Raise tolerance for render_camera tests ([662033c](https://github.com/langmm/canopy_factory/commit/662033cf1f6a9e35c598608790e79e9a82573ce5) by Meagan Lang).
- Add missing data for default maize when data not present (e.g. on GHA) Fix bug in scalar generation where np.float32 was returned instead of float Regenerate test data for for most recent openalea.lpy release Regenerate render camera test data for updated hothouse Change tests to use default maize canopy Add test option "--ignore-data" for local testing w/o data Use 'default' as default ID ([4b8c1ee](https://github.com/langmm/canopy_factory/commit/4b8c1ee71429b4cefdf1040f827eebae8e25a8d5) by Meagan Lang).
- Add linting to tests Update pyproject.toml and GHA tests workflow for pure python hothouse ([2bdb88a](https://github.com/langmm/canopy_factory/commit/2bdb88a2e995d3a0b7adf438e5a67f97c4f6b65d) by Meagan Lang).
- Add coverage & LAI calculations Add raytrace_stats to store top level raytrace properties (including limits) Fix bug in iteration over ID & data year Fix inset totals for animation Fix test for match query Fix totals for empty faces Added support for merging generator parameters ([68a69ac](https://github.com/langmm/canopy_factory/commit/68a69ac17a7461c5f76c662a53e9cd09c7bd86d1) by Meagan Lang).

### Fixed

- Fix rounding error in image resolution calculation Add test data for second raytrace timestep in totals to determine if the difference arises from the raytracer or from the totals calculation ([a5d7366](https://github.com/langmm/canopy_factory/commit/a5d7366013f72bdd3b71d413439e60b401e51cd8) by Meagan Lang).
- Fix indentation in GHA workflow ([f784fc3](https://github.com/langmm/canopy_factory/commit/f784fc3ca83f31e6d51f1d9440d915ae7a8c1fb4) by Meagan Lang).
- Fix docstrings for sphinx autodoc Use Python 3.12 for compatibility with lpy ([33eea6f](https://github.com/langmm/canopy_factory/commit/33eea6f89e66439ad747d511a214d06d94ba71e0) by Meagan Lang).
- Fix name for matched query Add CLI for controlling plotting parameters (line colors, styles, text weight, line width, figure resolution) Add iteration over canopy & periodic_canopy Standardize inclusion of data in CSV header comments via JSON Fix virtual & periodic canopy & regenerate test data Record timing statistics for raytrace operations Move comparison methods for testing into pytest fixtures ([9acb8f1](https://github.com/langmm/canopy_factory/commit/9acb8f153f6a6055fff47a623bff65772fbf4d5d) by Meagan Lang).
- Fix order of crop arguments & setting of default data year Disable yggdrasil installation Allow NodeElements to be defined Fix tests ([b0dc5b7](https://github.com/langmm/canopy_factory/commit/b0dc5b72bcc3a0465800e577e6c64d1207c64997) by Meagan Lang).

### Changed

- Change to using double precision for raytracer calculations ([7385f2c](https://github.com/langmm/canopy_factory/commit/7385f2c41cbdc65edaf35c629555653e804fd0ca) by Meagan Lang).

### Removed

- Remove license classifier Change python version used to build docs (No version of openalea.lpy for Python 3.14?) ([44e0cdd](https://github.com/langmm/canopy_factory/commit/44e0cdd38d3abbc62afbbb6edf616d061ff93c30) by Meagan Lang).
- Remove numpy pin ([635368f](https://github.com/langmm/canopy_factory/commit/635368f2e9dbf45f088662f52282e91db726bd36) by Meagan Lang).

## [v0.1.0](https://github.com/langmm/canopy_factory/releases/tag/v0.1.0) - 2026-01-09

<small>[Compare with first commit](https://github.com/langmm/canopy_factory/compare/863289d01b85a82a105f502003299154f42e4b68...v0.1.0)</small>

Initial package release.

### Summary of major features

- Procedural generation of crop canopies for maize, generic monocots, and generic dicots
- Ray tracing of generated canopies to calculate light interception throughout a day/season
- Growth of generated canopy throughout a season

### Added

- Added explicit classes for packaging arguments Updated name of wrapped storage for DictWrapper to avoid conflict with argument dest ([767fe0c](https://github.com/langmm/canopy_factory/commit/767fe0c2fb58d5514c083919b5bbd1067973c944) by Meagan Lang).
- Added tomato dummy class Fixed utilities for tracking & updating test data Added fruits Don't covert all crop generation parameters to CLI arguments for performance Allow generic iteration over multiple parameters (e.g. data_year) in addition to id ([d55ef84](https://github.com/langmm/canopy_factory/commit/d55ef845e384297fd86e5dfb58947916a4fb3470) by Meagan Lang).
- Added yggdrasil dependencies Added utilities for processing observation data into parameter data Added utilities for comparing observation data Updated dependencies to use yggdrasil-python-rapidjson directly ([fbcb74c](https://github.com/langmm/canopy_factory/commit/fbcb74c23eb9204ea3f6afb76eaff06ecc34cb96) by Meagan Lang).
- Added configuration file option Updated requirements to include ePhotosynthesis_C dependencies & workflow to install ePhotosynthesis_C Updated address for lpy build script ([b365610](https://github.com/langmm/canopy_factory/commit/b365610e955633e099da4ff86c03753b7629f8b5) by Meagan Lang).
- Added dicot base class & time dependency ([8b9dedc](https://github.com/langmm/canopy_factory/commit/8b9dedc88dbc7e20d1ab798dd4ad29df0b7b135e) by Meagan Lang).
- Added base class for iteration over times Added class for plotting the layout of simulated fields Added class for plotting the total intercepted light Moved all output into single directory Added support for periodic scenes w/ 2 as default ([c30989c](https://github.com/langmm/canopy_factory/commit/c30989ca916d576deab389f41a8d16f0c2d6c5da) by Meagan Lang).
- Add raytracer, render, & animate interfaces Add classes for managing parameters based on JSON schema Smooth geometry Attempt unfurling leaves, but generalized cylinder cannot change cross-sections ([ddea26a](https://github.com/langmm/canopy_factory/commit/ddea26a774af01b079584a0e98d89e4d65ed1263) by Meagan Lang).

### Fixed

- Fix permissions on build_ePhotosynthesis_C.sh ([6a9b284](https://github.com/langmm/canopy_factory/commit/6a9b2844153e2646f9ea4ec741aa6ed0cfe76498) by Meagan Lang).
- Fix git commands in build scripts to use https ([98f2f67](https://github.com/langmm/canopy_factory/commit/98f2f675da484f531068490248d089aaba49e419) by Meagan Lang).
- Fix syntax for updating the conda environment ([8185013](https://github.com/langmm/canopy_factory/commit/81850138b99481dfd95491283123222a19a739b8) by Meagan Lang).
- Fix README syntax Added images ([4679041](https://github.com/langmm/canopy_factory/commit/4679041f74ceb356075b1707e717fbf915f8e4cc) by Meagan Lang).

