import os
import pdb
import numpy as np
import itertools
import pprint
import copy
import time
from datetime import datetime
from yggdrasil_rapidjson import units
from canopy_factory import utils, arguments
from canopy_factory.utils import (
    cfg, get_class_registry,
    parse_quantity,
    cached_property, cached_args_property, readonly_cached_args_property,
    # Geometry
    scene2geom,
)
from canopy_factory.cli import (
    TaskBase, TemporalTaskBase,
    OptimizationTaskBase,
)
from canopy_factory.crops import GenerateTask, LayoutTask
from canopy_factory import light_sources
from canopy_factory.raytrace import hothouse


def generate_rays(ray_origins, ray_directions,
                  geom_format='obj', ray_color=(1.0, 0.0, 0.0),
                  ray_width=1.0, ray_length=10.0, arrow_width=2.0,
                  verbose=False):
    r"""Generate a set of rays for demonstration purposes.

    Args:
        ray_origins (np.ndarray): Positions that rays should start from.
        ray_directions (np.ndarray): Vectors describing how rays should
            point.
        geom_format (str, optional): Format that the rays geometries
            should be returned as.
        ray_color (tuple, optional): Set of RGB values in range 0.0 to 1.0
            designating the color that rays should be.
        ray_width (float, optional): Width of the ray stems.
        ray_length (float, optional): Length of the ray stems. This can
            also be provided as an array for each individual ray.
        arrow_width (float, optional): Width of the ray head.
        verbose (bool, optional): If True, log messages will be displayed
            for the generation process.

    """
    from openalea.lpy import Lsystem
    param = {
        'ORIGINS': ray_origins.astype(np.float64),
        'DIRECTIONS': ray_directions.astype(np.float64),
        'RAY_WIDTH': ray_width,
        'RAY_LENGTH': ray_length,
        'ARROW_WIDTH': arrow_width,
    }
    lsys = Lsystem(os.path.join(cfg['directories']['lpy'], 'rays.lpy'),
                   param)
    tree = lsys.axiom
    for i in range(2):
        tree = lsys.iterate(tree, 1)
    scene = lsys.sceneInterpretation(tree)
    mesh = scene2geom(
        scene, geom_format,
        color=ray_color, verbose=verbose,
    )
    if verbose:
        print('Finished generating rays')
    assert mesh.nface > 0
    return mesh


###################################################################
# TASKS
###################################################################


def unique(x):
    out = []
    for xx in x:
        if xx not in out:
            out.append(xx)
    return out


class RayTraceTask(TaskBase):
    r"""Class for running a ray tracer on a 3D canopy."""

    _query_options_adm = [
        'plantids', 'componentids', 'areas', 'height',
    ]
    _query_options_phy = [
        'flux_density', 'flux', 'hits', 'areas', 'height',
    ]
    _query_options = unique(_query_options_phy + _query_options_adm)
    _query_options_calc_prefix_per_plant = [
        'total_', 'average_', 'scene_average_', 'field_average_',
        'profile_density_', 'profile_', 'limits_',
    ]
    _query_options_calc_prefix = [
        'total_', 'average_', 'plant_average_', 'scene_average_',
        'field_average_',
        'profile_density_', 'profile_', 'limits_',
    ]
    _query_options_calc_prefix_depends = {
        'average_': ['total_'],
        'plant_average_': ['total_'],
        'scene_average_': ['total_'],
        'field_average_': ['total_'],
    }
    _query_options_depends = {
        'flux': ['flux_density', 'areas'],
    }
    _query_options_calc = [
        prefix + q for prefix, q in
        itertools.product(_query_options_calc_prefix, _query_options_phy)
    ]
    _query_options_other = [
        'compute_time',
    ]
    _query_options_stats = (
        _query_options_calc + _query_options_other
    ) + [
        'coverage', 'LAI',
    ]

    _name = 'raytrace'
    _output_info = {
        'raytrace': {
            'upstream': ['generate'],
            'ext': '.csv',
            'description': (
                'the query values for each face in the mesh'
            ),
            'composite_param': ['periodic_canopy'],
        },
        'traced_mesh': {
            'upstream': ['raytrace'],
            'description': (
                'the mesh with faces colored by a ray tracer result'
            ),
            'optional': True,
            'merge_all': True,
        },
        'raytrace_stats': {
            'upstream': ['raytrace'],
            'ext': '.json',
            'description': (
                'the top level statistics calculated from the ray trace'
            ),
        },
    }
    _external_tasks = {
        GenerateTask: {
            'exclude': ['age'],
            'modifications': {
                'mesh_format': {
                    'help': (
                        'Format that provided \"mesh\" is in or the '
                        'format that the generate mesh should be in if '
                        '\"mesh\" is not provided. If \"--mesh-format\" '
                        'is not provided, the file extension will be '
                        'used to determine the format'
                    ),
                },
                'mesh_units': {
                    'help': (
                        'Units that the provided \"mesh\" is in or '
                        'the units the generated mesh should be in if '
                        '\"mesh\" is not provided'
                    ),
                },
                'canopy': {
                    'default': 'unique',
                    'append_choices': ['virtual', 'virtual_single'],
                    'append_suffix_param': {
                        'prefix': 'canopy', 'title': True,
                        'cond': lambda x: (x.canopy != 'single'),
                    },
                },
                'plot_spacing': {
                    'append_suffix_param': {
                        'cond': lambda x: (
                            x.periodic_canopy or x.canopy != 'single'),
                    },
                },
                'plant_count': {
                    'append_suffix_param': {
                        'cond': lambda x: (x.canopy != 'single'),
                    },
                },
            },
            # Force virtual canopy between generate suffix and layout
            'increment_suffix_index': -3,
        },
        LayoutTask: {
            'include': [
                'periodic_canopy', 'periodic_canopy_count',
            ],
            'optional': True,
            'increment_suffix_index': -2,
        },
    }
    _arguments = [
        arguments.ClassSubparserArgumentDescription(
            'light_source',
            suffix_param={'noteq': 'sun'},
        ),
        arguments.CompositeArgumentDescription(
            'ray_color', 'color',
            description=(
                ' that should be used for rays when "--show-rays" '
                'is passed'
            ),
            defaults={
                'color': 'red',
            },
        ),
        arguments.CompositeArgumentDescription(
            'highlight_color', 'color',
            description=(
                'that should be used for highlighted faces if '
                '"--highlight" is passed'
            ),
            defaults={
                'color': 'magenta',
            },
        ),
        (('--raytracer', ), {
            'type': str, 'default': 'hothouse',
            'choices': list(get_class_registry().keys('raytracer')),
            'help': 'Name of the ray tracer that should be used.',
            'suffix_param': {'noteq': 'hothouse'},
        }),
        (('--separate-plants', ), {
            'action': 'store_true',
            'help': 'Track plants as separate scene components.',
        }),
        (('--separate-components', ), {
            'action': 'store_true',
            'help': ('Track plant components (e.g. leaf, stem) as '
                     'separate scene components.'),
        }),
        (('--nrays', ), {
            'type': int, 'default': 512,
            'help': ('Number of rays that should be cast along each '
                     'dimension'),
            'suffix_param': {'cond': True},
        }),
        (('--multibounce', ), {
            'action': 'store_true',
            'help': ('Include multiple bounces when performing the '
                     'trace.'),
            'suffix_param': {'value': 'multibounce'},
        }),
        (('--any-direction', ), {
            'action': 'store_true',
            'help': ('Allow light to be deposited by the ray tracer '
                     'from any direction relative to the surface. If '
                     'not set, only ray intercepting a surface from '
                     'the \"top\" will be counted.'),
            'suffix_param': {'value': 'anydirection'},
        }),
        (('--query', ), {
            'type': str, 'choices': _query_options,
            'default': 'flux',
            'help': ('Name of the raytracer query result that should '
                     'be used to color the traced mesh if '
                     '"--output-traced-mesh" is passed. '
                     '\"flux\" uses the intercepted flux density for '
                     'each triangle in the mesh, \"hits\" uses '
                     'the number of rays that hit each face in the '
                     'mesh, \"areas\" uses the area of each face, '
                     '\"plantids\" uses the IDs of the plant each face '
                     'belongs to, and \"componentids\" uses the IDs '
                     'of the plant architecture component.'),
            'suffix_param': {
                'cond': True,
                'skip_outputs': ['raytrace', 'raytrace_stats'],
            },
        }),
        (('--show-rays', ), {
            'action': "store_true",
            'help': ('Show the rays in the generated mesh if '
                     '"--output-traced-mesh" is passed.'),
            'suffix_param': {
                'value': 'rays',
                'skip_outputs': ['raytrace', 'raytrace_stats'],
            },
        }),
        (('--highlight', ), {
            'type': str, 'choices': ['min', 'max'],
            'help': ('Highlight the face with the \"min\" or \"max\" '
                     'query value in the resulting (only valid if '
                     '"--output-traced-mesh" is passed).'),
            'suffix_param': {
                'prefix': 'highlight',
                'title': True,
                'skip_outputs': ['raytrace', 'raytrace_stats'],
            },
        }),
        (('--ray-width', ), {
            'default': 1.0, 'units': 'cm',
            'units_arg': 'mesh_units',
            'help': 'Width of rays drawn when "--show-rays" is passed.',
        }),
        (('--ray-length', ), {
            'default': 10.0, 'units': 'cm',
            'units_arg': 'mesh_units',
            'help': ('Length of rays drawn when "--show-rays" is '
                     'passed. A negative value will cause the distance '
                     'to the scene to be used for the ray length.'),
        }),
        (('--arrow-width', ), {
            'default': 2.0, 'units': 'cm',
            'units_arg': 'mesh_units',
            'help': ('Width of arrows of rays drawn when "--show-rays" '
                     'is passed.'),
        }),
        arguments.CompositeArgumentDescription(
            'traced_mesh_colormap', 'colormap',
            description=(
                'that should be used for query values '
                'if "--output-traced-mesh" is passed'
            ),
            defaults={
                'colormap': 'YlGn_r',
            },
        ),
    ]

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if args.id == 'all' and name != 'traced_mesh':
            return
        super(RayTraceTask, cls)._write_output(args, name, fname, output)

    @property
    def verbose(self):
        r"""bool: Turn on log messages."""
        return self.args.verbose

    @readonly_cached_args_property
    def raytracer(self):
        r"""RayTracerBase: Ray tracer."""
        print("Re-creating ray tracer", self.args.time.value,
              self.args.age.value)
        assert self.geometryids is not None
        return get_class_registry().get(
            'raytracer', self.args.raytracer)(
                self.args, self.mesh,
                geometryids=self.geometryids)

    @cached_args_property
    def nvirtual(self):
        if not self.args.canopy.startswith('virtual'):
            return 0
        return np.prod(self.args.virtual_canopy_count_array) - 1

    @cached_property
    def scene_area(self):
        r"""units.Quantity: Area that the scene covers."""
        if self.args.canopy != 'single':
            return self.args.plot_width * self.args.plot_length
        return self.plant_area

    @cached_property
    def planting_density(self):
        r"""units.Quantity: Planting density."""
        return (self.args.ncols * self.args.nrows) / self.scene_area

    @cached_property
    def plant_area(self):
        r"""units.Quantity: Area that the scene covers."""
        return self.args.row_spacing * self.args.plant_spacing

    @cached_property
    def field_area(self):
        r"""units.Quantity: Area that the scene covers."""
        assert len(self.raytracer.field_scene_model.dim) == 2
        return utils.safe_op(
            np.prod, self.raytracer.field_scene_model.dim
        )

    @cached_property
    def mesh(self):
        r"""ObjDict: Mesh that will be ray traced."""
        self.args.output_generate.assert_age_in_name(self.args)
        return self.get_output('generate')

    @cached_property
    def geometryids(self):
        return self.get_output('geometryids')

    @cached_property
    def component_order(self):
        return self.geometryids['HEADER_JSON']['component_order']

    @cached_property
    def componentids(self):
        r"""np.ndarray: Component IDs for each face in the mesh."""
        return self.geometryids['componentids']

    @cached_property
    def plantids(self):
        r"""np.ndarray: Plant IDs for each face in the mesh."""
        return self.geometryids['plantids']

    @classmethod
    def adjust_args_internal(cls, args, skip=None, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            skip (list, optional): Set of arguments to skip.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        if skip is None:
            skip = []
        args.include_units = True
        if not GenerateTask.mesh_generated(args):
            args.periodic_canopy = False
            args.canopy = 'external'
        if args.overwrite_raytrace:
            args.overwrite_raytrace_stats = True
        super(RayTraceTask, cls).adjust_args_internal(
            args, skip=skip, **kwargs)

    @classmethod
    def _output_ext(cls, args, name, wildcards=None, skipped=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skipped (list, optional): List of arguments that should be
                skipped in the generated output file name.

        Returns:
            str: Output file extension.

        """
        if name == 'traced_mesh':
            return cls._outputs_external['generate']._output_ext(
                args, 'generate', wildcards=wildcards, skipped=skipped)
        return None

    @classmethod
    def extract_query(cls, query_values, query):
        r"""Extract a query value, scaling as necessary.

        Args:
            query_values (dict): Query values for each face.
            query (str): Query value to return.

        Returns:
            np.ndarray: Value for each face.

        """
        if query in query_values:
            return query_values[query]
        elif query == 'flux':
            return query_values['flux_density'] * query_values['areas']
        elif query in cls._query_options_other:
            return query_values['HEADER_JSON'][query]
        raise NotImplementedError(query)

    def _fill_in_admin_queries(self, values):
        values = {k: v for k, v in values.items()}
        for k in self._query_options_adm:
            if k in values:
                continue
            out = self.geometryids[k]
            if self.args.canopy == 'virtual':
                out = [out]
                if k == 'plantids':
                    maxid = out[0].max() + 1
                    for ivert in range(1, self.nvirtual + 1):
                        out.append(out[0] + ivert * maxid)
                else:
                    for ivert in range(1, self.nvirtual + 1):
                        out.append(out[0])
                out = np.hstack(out)
            values[k] = out
        return values

    @classmethod
    def _fill_in_calculated_queries(cls, values, query):
        for k in cls._query_options_depends.keys():
            if k in values:
                continue
            if (((isinstance(query, list) and k in query)
                 or query == k)):
                values[k] = cls.extract_query(values, k)

    @classmethod
    def _check_plantids(cls, values, per_plant):
        plantids_unique = np.unique(values['plantids'])
        if not isinstance(per_plant, bool):
            if ((len(plantids_unique) != per_plant
                 and len(plantids_unique) == 0)):
                plantids_unique = range(per_plant)
            else:
                if len(plantids_unique) != per_plant:
                    print(len(plantids_unique), per_plant)
                    pdb.set_trace()
                assert len(plantids_unique) == per_plant
        return plantids_unique

    @classmethod
    def join_query_stats(cls, values, query=None, per_plant=False):
        r"""Concatenate query statistics as an array.

        Args:
            values (list): Set of query results.
            query (str, optional): Query statistic in values. If not
                set, values must be a list of stats dictionaries and
                the result will be a dictionary of the concatenated
                values for each stat.
            per_plant (bool, optional): If True and query supports per-
                plant statistics, the values are dictionaries of per-
                plant stats.

        Returns:
            np.ndarray: Concatenated stats.

        """
        if query is None:
            assert isinstance(values[0], dict)
            out = {}
            for k in values[0].keys():
                out[k] = cls.join_query_stats(
                    [v[k] for v in values], k, per_plant=per_plant,
                )
            return out
        if not query.startswith(
                tuple(cls._query_options_calc_prefix_per_plant)):
            per_plant = False

        def create_dst(v):
            if isinstance(v, dict):
                return {
                    k: create_dst(vv) for k, vv in v.items()
                }
            return []

        if isinstance(values[0], dict):
            out = {
                k: np.vstack([
                    x[k] if not (isinstance(x[k], np.ndarray)
                                 and x[k].shape)
                    else x[k].T
                    for x in values
                ]) for k in values[0].keys()
            }
            for k, v in values[0].items():
                if isinstance(v, units.QuantityArray):
                    out[k] = units.QuantityArray(out[k], v.units)
        else:
            out = np.vstack([
                x if not (isinstance(x, np.ndarray) and x.shape)
                else x.T
                for x in values
            ])
            if isinstance(values[0], units.QuantityArray):
                out = units.QuantityArray(out, values[0].units)
        return out

    @classmethod
    def extract_query_stat(cls, method, values, query, per_plant=False,
                           include_base=False, plantid=None,
                           scale_total=None, **kwargs):
        r"""Calculate a statistic of the query.

        Args:
            method (callable): Function to call to calculate stats.
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return stat(s) for.
                If a list is provided, a dictionary mapping from query
                name to value will be returned.
            per_plant (bool, optional): If True, the query values should
                be profiled for each plant. If an int is provided,
                the number of unique plant IDs should match the provided
                value.
            include_base (bool, optional): If True, include the stats
                for the whole scene when per_plant is True.
            plantid (int, optional): Plant ID to extract the profile for.
                None indicates the entire canopy.
            scale_total (float, optional): Scale factor to apply to the
                total for all plants.
            **kwargs: Additional keyword arguments are passed to the
                provided method.

        Returns:
            dict: Stats for the specified query(s).

        """
        cls._fill_in_calculated_queries(values, query)
        if isinstance(query, list):
            return {
                k: cls.extract_query_stat(
                    method, values, k, per_plant=per_plant,
                    include_base=include_base, plantid=plantid,
                    scale_total=scale_total, **kwargs
                )
                for k in query
            }
        if per_plant:
            assert plantid is None
            plantids_unique = cls._check_plantids(values, per_plant)
            out = {}
            for i in plantids_unique:
                out[str(i)] = cls.extract_query_stat(
                    method, values, query, plantid=i,
                    scale_total=scale_total, **kwargs)
            if include_base:
                out['total'] = cls.extract_query_stat(
                    method, values, query,
                    scale_total=scale_total, **kwargs)
            return out
        if plantid is not None:
            idx = (values['plantids'] == plantid)
            values = {k: values[k][idx] for k in values.keys()
                      if k != 'HEADER_JSON'}
        out = method(values, query, **kwargs)
        if plantid is None and scale_total is not None:
            out *= scale_total
        return out

    @classmethod
    def extract_query_average(cls, values, query, **kwargs):
        r"""Calculate average of total per plant.

        Args:
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return total(s) for. If
                a list is provided, a dictionary mapping from query name
                to value will be returned.
            **kwargs: Additional keyword arguments are passed to
                extract_query_stat.

        Returns:
            units.Quantity: Average for the specified query across plants.

        """
        def func(values, query):
            return utils.safe_op(np.mean, values[query],
                                 value_on_empty=0.0)

        return cls.extract_query_stat(func, values, query, **kwargs)

    @classmethod
    def extract_query_total(cls, values, query, **kwargs):
        r"""Calculate sums over one query options.

        Args:
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return total(s) for. If
                a list is provided, a dictionary mapping from query name
                to value will be returned.
            **kwargs: Additional keyword arguments are passed to
                extract_query_stat.

        Returns:
            units.Quantity: Total for the specified query.

        """
        def func(values, query):
            return utils.safe_op(np.sum, values[query],
                                 value_on_empty=0.0)

        return cls.extract_query_stat(func, values, query, **kwargs)

    @classmethod
    def extract_query_profile(cls, values, query, **kwargs):
        r"""Calculate the profile of the query over canopy height.

        Args:
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return profile(s) for.
                If a list is provided, a dictionary mapping from query
                name to value will be returned.
            nbins (int, optional): Number of bins to use in the profile.
            density (bool, optional): If True, the profile will be a
                density normalized by the number of values in each bin.
            **kwargs: Additional keyword arguments are passed to
                extract_query_stat.

        Returns:
            units.QuantityArray: Profile for the specified query.

        """

        def func(values, query, nbins=20, density=False):
            x = values[query]
            x_units = None
            if isinstance(x, units.QuantityArray):
                x_units = x.units
                x = x.value
            heights = values['height']
            heights_units = None
            if isinstance(heights, units.QuantityArray):
                heights_units = heights.units
                heights = heights.value
            if len(heights) == 0:
                hist = np.zeros((nbins, ), dtype=x.dtype)
                bins = np.linspace(0, 1, nbins, dtype=heights.dtype)
            else:
                hist, bins = np.histogram(heights, weights=x,
                                          bins=nbins, density=density)
                bins = (bins[:-1] + bins[1:]) / 2
            if heights_units is not None:
                bins = parse_quantity(bins, heights_units)
            if x_units is not None:
                hist = parse_quantity(hist, x_units)
            return {'bins': bins, 'hist': hist}

        return cls.extract_query_stat(func, values, query, **kwargs)

    @classmethod
    def extract_query_limits(cls, values, query, **kwargs):
        r"""Calculate the limits of the query.

        Args:
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return limit(s) for.
                If a list is provided, a dictionary mapping from query
                name to value will be returned.
            **kwargs: Additional keyword arguments are passed to
                extract_query_stat.

        Returns:
            dict: Limits for the specified query.

        """

        def func(values, query):
            x = values[query]
            x_units = None
            if isinstance(x, units.QuantityArray):
                x_units = x.units
                x = x.value
            if len(x) == 0:
                out = {
                    'linear': np.array([np.nan, np.nan]),
                    'log': np.array([np.nan, np.nan]),
                }
            elif (x == 0).all():
                out = {
                    'linear': np.array([x[x >= 0].min(), x.max()]),
                    'log': np.array([np.nan, np.nan]),
                }
            else:
                out = {
                    'linear': np.array([x[x >= 0].min(), x.max()]),
                    'log': np.array([x[x > 0].min(), x.max()]),
                }
            if x_units is not None:
                out = {k: parse_quantity(v, x_units)
                       for k, v in out.items()}
            return out

        return cls.extract_query_stat(func, values, query, **kwargs)

    def calculate_query(self, query, prevent_output=False,
                        per_plant=False):
        r"""Calculate a query value for the scene.

        Args:
            query (str): Query value to return.
            prevent_output (bool, optional): If True, don't output
                raytracer results if they don't already exist.
            per_plant (bool, optional): If True, the query values should
                also be totaled for each plant. If an int is provided,
                the number of unique plant IDs should match the provided
                value. Only used for "total_*" query names.

        Returns:
            units.Quantity: Calculated query value.

        """
        query_sorted = {
            k: [] for k in self._query_options_calc_prefix + ['']
        }

        def get_base(x):
            for prefix in self._query_options_calc_prefix:
                if x.startswith(prefix):
                    out = x.split(prefix, 1)[-1]
                    query_sorted[prefix].append(out)
                    prefix_req = (
                        self._query_options_calc_prefix_depends.get(
                            prefix, [])
                    )
                    for iprefix_req in prefix_req:
                        query_sorted[iprefix_req].append(out)
                    return out
            else:
                query_sorted.setdefault('', [])
                query_sorted[''].append(x)
                return x

        singular = (not isinstance(query, list))
        if singular:
            query = [query]
        query_base = set(get_base(k) for k in query)
        if (not prevent_output) or self.output_exists():
            values = self.get_output('raytrace')
        else:
            values = self.raytrace_scene(query=list(query_base))
        values = self._fill_in_admin_queries(values)
        out = {}
        if query_sorted['plant_average_'] and per_plant is False:
            per_plant = True
        for prefix, kquery in query_sorted.items():
            if not kquery:
                continue
            multiple_values = False
            if prefix == 'total_':
                scale_total = 1
                if self.args.canopy == 'virtual_single':
                    scale_total = self.args.nrows * self.args.ncols
                iout = self.extract_query_total(
                    values, kquery,
                    per_plant=per_plant, include_base=True,
                    scale_total=scale_total,
                )
            elif prefix == 'average_':
                iout = self.extract_query_average(
                    values, kquery,
                    per_plant=per_plant, include_base=True,
                )
            elif prefix == 'plant_average_':
                iout = {}
                for k in kquery:
                    ktotal = f'total_{k}'
                    assert ktotal in out and per_plant
                    iout[k] = utils.safe_op(
                        np.mean,
                        [
                            vv for kk, vv in out[ktotal].items()
                            if kk != 'total'
                        ],
                        value_on_empty=0.0,
                    )
            elif prefix == 'scene_average_':
                iout = {}
                for k in kquery:
                    ktotal = f'total_{k}'
                    assert ktotal in out
                    if per_plant:
                        iout[k] = {
                            kk: vv / (self.scene_area if kk == 'total'
                                      else self.plant_area)
                            for kk, vv in out[ktotal].items()
                        }
                    else:
                        iout[k] = out[ktotal] / self.scene_area
            elif prefix == 'field_average_':
                iout = {}
                for k in kquery:
                    ktotal = f'total_{k}'
                    assert ktotal in out
                    if per_plant:
                        # TODO: Get the actual bounds for the plant
                        iout[k] = {
                            kk: vv / (self.field_area if kk == 'total'
                                      else self.plant_area)
                            for kk, vv in out[ktotal].items()
                        }
                    else:
                        iout[k] = out[ktotal] / self.field_area
            elif prefix.startswith('profile_'):
                iout = self.extract_query_profile(
                    values, kquery,
                    density=(prefix == 'profile_density_'),
                    per_plant=per_plant, include_base=True,
                )
                multiple_values = 'hist'
            elif prefix == 'limits_':
                iout = self.extract_query_limits(
                    values, kquery,
                    per_plant=per_plant, include_base=True,
                )
                multiple_values = True
            elif prefix == '':
                iout = {k: self.extract_query(values, k)
                        for k in kquery}
            else:
                raise NotImplementedError(prefix)
            if multiple_values:
                for k, v in iout.items():
                    if per_plant:
                        for ksrc in v['0'].keys():
                            kdst = (
                                f'{prefix}{k}' if ksrc == multiple_values
                                else f'{prefix}{k}_{ksrc}'
                            )
                            out[kdst] = {
                                kk: vv[ksrc] for kk, vv in v.items()
                            }
                    else:
                        for ksrc in v.keys():
                            kdst = (
                                f'{prefix}{k}' if ksrc == multiple_values
                                else f'{prefix}{k}_{ksrc}'
                            )
                            out[kdst] = v[ksrc]
            else:
                for k, v in iout.items():
                    out[f'{prefix}{k}'] = v
        if singular:
            return out[query[0]]
        return out

    @classmethod
    def update_color_limits(cls, query, cmap, limits, force=False):
        r"""Set the minimum and maximum for color mapping.

        Args:
            query (str): Raytrace target that limits should be taken
                from.
            cmap (ColorMapArgument): Colormap argument information.
            limits (dict): Set of calculated limits for the query.
            force (bool, optional): If True, overwrite any existing
                arguments for the color limits.

        Returns:
            bool: True if the limits were generated.

        """
        if (not force) and cmap.limits_defined:
            return False
        out = False
        if force or cmap.vmin is None:
            cmap.vmin = limits[0]
        if force or cmap.vmax is None:
            cmap.vmax = limits[1]
        cls.log_class(f'LIMITS[{cls._name}, {cmap.name}]: '
                      f'{cmap.vmin}, {cmap.vmax}')
        return out

    def _color_scene(self, query):
        r"""Color the geometry based on the raytracer results.

        Args:
            query (str): Raytrace target that should be used to color
                the scene.

        """
        query_values = self.get_output('raytrace')
        if self.args.show_rays:
            inst = self.run_iteration(
                args_overwrite={'show_rays': False,
                                'output_traced_mesh': True},
            )
            mesh = inst.get_output('traced_mesh')
            rays = generate_rays(inst.raytracer.ray_origins,
                                 inst.raytracer.ray_directions,
                                 ray_length=inst.raytracer.ray_lengths,
                                 geom_format=type(mesh),
                                 ray_color=self.args.ray_color.value,
                                 ray_width=self.args.ray_width.value,
                                 arrow_width=self.args.arrow_width.value)
            mesh.append(rays)
            return mesh
        cmap = self.args.traced_mesh_colormap
        if not cmap.limits_defined:
            self.update_color_limits(
                query, cmap, self.limits[query][cmap.scaling],
            )
        mesh = self.mesh
        face_values = self.extract_query(query_values, query)
        if isinstance(face_values, units.QuantityArray):
            face_values = np.array(face_values)
        vertex_values = self.raytracer.face2vertex(
            face_values, method='deposit')
        vertex_colors = utils.apply_color_map(
            vertex_values,
            color_map=cmap.colormap,
            vmin=cmap.vmin,
            vmax=cmap.vmax,
            scaling=cmap.scaling,
            highlight=self.args.highlight,
            highlight_color=self.args.highlight_color.value,
        )
        mesh.add_colors('vertex', vertex_colors)
        return mesh

    def raytrace_scene(self, query=None):
        r"""Run the ray tracer on the selected geometry.

        Args:
            query (str, optional): Name of the field that should be
                calculated.

        Returns:
            dict: Dictionary of ray tracer queries.

        """
        if query is None:
            query = self._query_options + self._query_options_other
        if isinstance(query, list):
            values = {}
            if 'compute_time' in query:
                t0 = time.time()
            for k, kreq in self._query_options_depends.items():
                if k in query:
                    query = query + [kkreq for kkreq in kreq
                                     if kkreq not in query]
            for k in query:
                if k in (list(self._query_options_depends.keys())
                         + self._query_options_other):
                    continue
                values[k] = self.raytrace_scene(query=k)
            if 'flux' in query:
                values['flux'] = self.extract_query(values, 'flux')
            if 'compute_time' in query:
                t1 = time.time()
                values['HEADER_JSON'] = {
                    'compute_time': t1 - t0,
                }
            return values
        if query in self._query_options_depends:
            values = self.raytrace_scene(
                query=self._query_options_depends[query])
            return self.extract_query(values, query)
        return self.raytracer.raytrace(query)

    def _canopy_profile(self, query):
        r"""Calculate the profile of a query value at different positions
        in the canopy.

        Args:
            query (str, list): Query value(s) to compute the profile for.

        """
        raise NotImplementedError

    def _generate_stats(self):
        r"""Generate top level raytrace stats.

        Returns:
            dict: Top level statistics.

        """
        if self.args.canopy == 'virtual_single':
            per_plant = 1
        else:
            per_plant = (self.args.nrows * self.args.ncols)
        query = (
            self._query_options_calc
            + self._query_options_other
        )
        output = self.calculate_query(query, per_plant=per_plant)
        output['coverage'] = self.raytracer.coverage()
        leaf_area = self.geometryids['HEADER_JSON']['leaf_area']
        if not self.args.canopy.startswith('virtual'):
            leaf_area = leaf_area / (self.args.nrows * self.args.ncols)
        output['LAI'] = leaf_area * self.planting_density
        return output

    @classmethod
    def _generate_limits_class(cls, args, limits_args_base=None,
                               **kwargs):
        r"""Generate limits for the date in question.

        Args:
            args (argparse.Namespace): Parsed arguments.
            limits_args_base (dict): Arguments to use to generate base
                limits.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            dict: Limits.

        """
        if limits_args_base is None:
            limits_args_base = cls.limits_args_class(args, base=True)
        args_overwrite = dict(
            limits_args_base,
            **{
                'query': None,
                'planting_date': None,
                'output_raytrace_stats': True,
                'dont_write_raytrace_stats': False,
            }
        )
        optional_output = cls._output_names(
            args, include_external=True, exclude_required=True)
        for k in optional_output:
            if k == 'raytrace_stats':
                continue
            args_overwrite[f'output_{k}'] = False
        print(80 * '-')
        print("GENERATING LIMITS FOR BASE")
        pprint.pprint(args_overwrite)
        out = cls.run_iteration_class(
            args, args_overwrite=args_overwrite,
            output_name='raytrace_stats',
            **kwargs
        )
        print("END GENERATING LIMITS FOR BASE")
        print(80 * '-')
        return out

    @cached_property
    def limits(self):
        r"""Generate limits for the date in question.

        Returns:
            dict: Limits.

        """
        limits_args = self.limits_args_class(self.args)
        limits_args_base = self.limits_args_class(self.args, base=True)
        if limits_args != limits_args_base:
            stats = self._generate_limits_class(
                self.args, limits_args_base=limits_args_base,
            )
        else:
            stats = self.get_output('raytrace_stats')
        out = {}
        for k, v in stats.items():
            if not k.startswith('limits_'):
                continue
            k = k.split('limits_', 1)[-1]
            query, scaling = k.rsplit('_', 1)
            out.setdefault(query, {})
            out[query][scaling] = v['total']
        return out

    @classmethod
    def is_limits_base_class(cls, args):
        r"""Check if the current arguments describe the set used to
        generate the base limits.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the current arguments are what is required
                for calculating limits.

        """
        limits_args = cls.limits_args_class(args)
        limits_args_base = cls.limits_args_class(args, base=True)
        return (limits_args == limits_args_base)

    @classmethod
    def limits_args_class(cls, args, base=False):
        r"""Get the arguments controling limits.

        Args:
            args (argparse.Namespace): Parsed arguments.
            base (bool, optional): If True, return the arguments that
                should be used to generate the limits rather than the
                current arguments.

        Returns:
            dict: Set of arguments controlling limits.

        """
        if base:
            out = {
                'time': 'noon',
                'date': None,
                'doy': 'summer_solstice',
            }
        else:
            out = {
                'time': args.time.solar_time_string,
                'date': None,
                'doy': args.time.solar_date_string,
            }
        if not GenerateTask.mesh_generated(args):
            return out
        if base:
            out.update(
                age='maturity',
                **GenerateTask.base_param_class(args)
            )
        else:
            out.update(
                id=args.id,
                data_year=args.data_year,
                age=args.time.crop_age_string,
            )
        return out

    def _merge_output(self, name, output, merged_param):
        r"""Merge the output for multiple sets of parameters values.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from tuples of parameter values to
               the output for the parameter values.
            merged_param (tuple): Names of the parameters that are being
                merged (the parameters specified by the tuple keys in
                output).

        Returns:
            object: Generated output.

        """
        if name != 'traced_mesh':
            return super(RayTraceTask, self)._merge_output(
                name, output, merged_param)
        return GenerateTask.merge_mesh(self.args, list(output.values()))

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id != 'all'
        if name == 'raytrace':
            out = self.raytrace_scene()
            # Prevent saving redundant info store in geometryids
            for k in self._query_options_adm:
                out.pop(k, None)
            return out
        elif name == 'traced_mesh':
            return self._color_scene(self.args.query)
        elif name == 'raytrace_stats':
            return self._generate_stats()
        super(RayTraceTask, self)._generate_output(name)


class RenderTask(TaskBase):
    r"""Class for rendering a 3D canopy."""

    _name = 'render'
    _output_info = {
        'render': {
            'ext': '.png',
            'base_output': 'raytrace',
            'description': 'the rendered image',
            'merge_all': True,
        },
        'render_camera': {
            'ext': '.json',
            'base_output': 'generate',
            'description': 'camera properties',
            'optional': True,
        },
    }
    _external_tasks = {
        RayTraceTask: {
            'exclude': [
                'show_rays', 'ray_color', 'ray_width', 'ray_length',
                'arrow_width', 'highlight', 'highlight_color',
                'traced_mesh_colormap',
            ],
            'modifications': {
                'query': {
                    'append_suffix_outputs': ['render'],
                },
            },
        },
    }
    _arguments = [
        (('--camera-direction', ), {
            'type': str,
            'help': ('Direction that camera should face. If not '
                     'provided, the camera will point to the center of '
                     'the scene from its location.'),
            'suffix_param': {},
        }),
        (('--camera-location', ), {
            'type': str,
            'help': ('Location of the camera. If not provided, one will '
                     'be determined that captures the entire scene from '
                     'the provided camera direction. If a direction is '
                     'also not provided, the camera will be centered '
                     'on the center of the scene facing down, '
                     'southeast at a distance that captures the entire '
                     'scene. If \"maturity\" is specified, the location '
                     'will be set for the mature plant (only valid for '
                     'generated meshes).'),
            'suffix_param': {'prefix': 'from_'},
        }),
        arguments.CompositeArgumentDescription(
            'scene_age', 'age',
            description=(
                'that the camera position should be calculated for '
                '(only valid for generated meshes)'
            ),
            ignore=['planting_date'],
            optional=True,
            suffix_param={}
        ),
        (('--camera-type', ), {
            'type': str,
            'choices': ['projection', 'orthographic'],  # 'spherical'],
            'default': 'projection',
            'help': ('Type of camera that should be used to render the '
                     'scene'),
            'suffix_param': {'noteq': 'projection'},
        }),
        arguments.DimensionArgumentDescription([
            (('--camera-fov-width', ), {
                'units': 'degrees', 'default': 45.0,
                'help': (
                    'Angular width of the camera\'s field of view (in '
                    'degrees) for a projection camera.'
                ),
                'suffix_param': {'conv': int},
            }),
            (('--camera-fov-height', ), {
                'units': 'degrees', 'default': 45.0,
                'help': (
                    'Angular height of the camera\'s field of view (in '
                    'degrees) for a projection camera.'
                ),
                'suffix_param': {'conv': int},
            }),
        ], name='camera_fov_dimensions', suffix_param={
            'cond': lambda x: (x.camera_type == 'projection'),
        }),
        (('--camera-up', ), {
            'type': str,
            'help': ('Up direction for the camera. If not provided, the '
                     'up direction for the scene will be assumed.'),
            'suffix_param': {'prefix': 'up', 'sep': 'x'},
        }),
        arguments.CompositeArgumentDescription(
            'background_color', 'color',
            description='that should be used for the scene',
            defaults={'color': 'transparent'},
            suffix_param={'noteq': 'transparent'},
        ),
        arguments.DimensionArgumentDescription([
            (('--image-nx', ), {
                'type': int,
                'help': (
                    'Number of pixels for the rendered image in the '
                    'horizontal direction. If not provided, but '
                    '--image-ny is provided, the value for --image-nx '
                    'will be determined from --image-ny by assuming '
                    'a constant resolution in both directions. If '
                    'neither are provided, --image-ny defaults to '
                    '1024.'
                ),
                'suffix_param': {},
            }),
            (('--image-ny', ), {
                'type': int,
                'help': (
                    'Number of pixels for the rendered image in the '
                    'vertical direction. If not provided, but '
                    '--image-nx is provided, the value for --image-ny '
                    'will be determined from --image-nx by assuming '
                    'a constant resolution in both directions. If '
                    'neither are provided, --image-ny defaults to '
                    '1024.'
                ),
                'suffix_param': {},
            }),
        ], name='resolution_dimensions', suffix_param={}),
        arguments.DimensionArgumentDescription([
            (('--image-width', ), {
                'units': 'cm',
                'units_arg': 'mesh_units',
                'help': (
                    'Width of the image (in cm). If not provided, '
                    'the width will be set based on the camera '
                    'position and type such that the entire scene '
                    'is captured.'
                ),
                'suffix_param': {'conv': int},
            }),
            (('--image-height', ), {
                'units': 'cm',
                'units_arg': 'mesh_units',
                'help': (
                    'Height of the image (in cm). If not provided, '
                    'the height will be set based on the camera '
                    'position and type such that the entire scene '
                    'is captured.'
                ),
                'suffix_param': {'conv': int},
            }),
        ], name='image_dimensions', suffix_param={
            'suffix': 'cm',
        }),
        (('--resolution', ), {
            'units': 'cm**-1',  # 'default': 5,
            'help': ('Resolution that the scene should be rendered with '
                     'in pixels per centimeter. If provided, any '
                     'values provided for --image-nx and --image-ny '
                     'will be ignored. If not provided, the resolution '
                     'in each direction will be determined by '
                     '--image-nx and --image-ny.'),
            'suffix_param': {
                'conv': int,
                'suffix': 'percm',
            },
        }),
        (('--scene-mins', ), {
            'help': 'Minimum extent of scene along each dimension',
            'suffix_param': {
                'sep': 'x',
                'prefix': 'SCMIN',
            },
        }),
        (('--scene-maxs', ), {
            'help': 'Maximum extent of scene along each dimension',
            'suffix_param': {
                'sep': 'x',
                'prefix': 'SCMAX',
            },
        }),
        arguments.CompositeArgumentDescription(
            'render_colormap', 'colormap',
            description=(
                'that should be used for query values '
                'if "--output-render" is passed'
            ),
            defaults={
                'colormap': 'viridis',
                'colormap_scaling': 'log',
            },
        ),
    ]

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        args.show_rays = False
        if ((args.camera_direction is None
             and args.camera_location is None)):
            args.camera_direction = 'downnortheast'
        super(RenderTask, cls).adjust_args_internal(args, **kwargs)
        if not cls.is_camera_base_class(args):
            args.dont_write_render_camera = True

    @classmethod
    def adjust_args_step(cls, args, vary):
        r"""Adjust the parsed arguments before the class is run in a
        temporal iteration.

        Args:
            args (argparse.Namespace): Parsed arguments.
            vary (str): Name of argument being varied.

        """
        if vary == 'time':
            if args.start_time.age == args.stop_time.age:
                return
            args.scene_age = args.stop_age

    @property
    def raytracer(self):
        r"""RayTracerBase: Ray tracer."""
        return self.external_tasks['raytrace'].raytracer

    def _render_scene(self, query):
        r"""Render the scene using a ray tracer.

        Args:
            query (str): Raytrace target that should be used to color
                the scene.

        Returns:
            np.ndarray: Pixel color data.

        """
        query_values = self.get_output('raytrace')
        cmap = self.args.render_colormap
        if not cmap.limits_defined:
            RayTraceTask.update_color_limits(
                query, cmap,
                self.external_tasks['raytrace'].limits[query][cmap.scaling],
            )
        face_values = RayTraceTask.extract_query(query_values, query)
        if isinstance(face_values, units.QuantityArray):
            face_values = np.array(face_values)
        pixel_values = self.raytracer.render(face_values)
        if isinstance(pixel_values, units.QuantityArray):
            pixel_values = np.array(pixel_values)
        # self.set_output('pixel_values', pixel_values)
        pixel_values = (pixel_values.T)[::-1, :]
        image = utils.apply_color_map(
            pixel_values,
            color_map=cmap.colormap,
            vmin=cmap.vmin,
            vmax=cmap.vmax,
            scaling=cmap.scaling,
            highlight=(pixel_values < 0),
            highlight_color=self.args.background_color.value,
            include_alpha=(len(self.args.background_color.value) == 4)
        )
        return image

    @classmethod
    def is_camera_base_class(cls, args):
        r"""Check if the current arguments describe the set used to
        generate the base camera.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the current arguments are what is required
                for setting up the base camera.

        """
        camera_args = cls.camera_args_class(args)
        camera_args_base = cls.camera_args_class(args, base=True)
        return (camera_args == camera_args_base)

    @classmethod
    def camera_args_class(cls, args, base=False):
        r"""Get the arguments controling the camera.

        Args:
            args (argparse.Namespace): Parsed arguments.
            base (bool, optional): If True, return the arguments that
                should be used to generate the camera rather than the
                current arguments.

        Returns:
            dict: Set of arguments controlling the camera.

        """
        if not GenerateTask.mesh_generated(args):
            return {}
        if base:
            out = {
                'age': 'maturity',
                'scene_age': None,
            }
            if args.scene_age.args['age']:
                out['age'] = args.scene_age.args['age']
        else:
            out = {
                'age': args.time.args['age'],
                'scene_age': args.scene_age.args['age'],
            }
        out['canopy'] = args.canopy
        out['nrows'] = args.nrows
        out['ncols'] = args.ncols
        return out

    @classmethod
    def _generate_camera_class(cls, args, camera_args_base=None,
                               **kwargs):
        r"""Generate camera properties for a specific scene age.

        Args:
            args (argparse.Namespace): Parsed arguments.
            camera_args_base (dict): Arguments to use to generate base
                camera properties.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            dict: Camera properties.

        """
        if camera_args_base is None:
            camera_args_base = cls.camera_args_class(args, base=True)
        args_overwrite = dict(
            camera_args_base,
            **{
                'output_render': False,
                'output_render_camera': True,
                'dont_write_render_camera': False,
                'scene_mins': None,
                'scene_maxs': None,
                'time': None,
            }
        )
        optional_output = cls._output_names(
            args, include_external=True, exclude_required=True)
        for k in optional_output:
            if k == 'render_camera':
                continue
            args_overwrite[f'output_{k}'] = False
        print(80 * '-')
        print("GENERATING CAMERA PROPERTIES FOR BASE")
        pprint.pprint(args_overwrite)
        out = cls.run_iteration_class(
            args, args_overwrite=args_overwrite,
            output_name='render_camera',
            **kwargs
        )
        print("END GENERATING CAMERA PROPERTIES FOR BASE")
        print(80 * '-')
        return out

    def _generate_camera(self):
        r"""Generate camera properties for a specific scene age.

        Returns:
            dict: Camera properties.

        """
        camera_args = self.camera_args_class(self.args)
        camera_args_base = self.camera_args_class(self.args, base=True)
        if camera_args != camera_args_base:
            return self._generate_camera_class(
                self.args, camera_args_base=camera_args_base,
                cached_outputs=self._cached_outputs,
                cache_outputs=['render_camera'],
            )
        out = {}
        for k in ['camera_type', 'camera_fov_width',
                  'camera_fov_height']:
            out[k] = getattr(self.args, k)
        for k in ['camera_direction', 'camera_up', 'camera_location',
                  'image_nx', 'image_ny',
                  'image_width', 'image_height',
                  'resolution']:
            out[k] = getattr(self.raytracer, k)
        for k in ['mins', 'maxs']:
            out[f'scene_{k}'] = getattr(
                self.raytracer.virtual_scene_model, k)
        return out

    def _merge_output(self, name, output, merged_param):
        r"""Merge the output for multiple sets of parameters values.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from tuples of parameter values to
               the output for the parameter values.
            merged_param (tuple): Names of the parameters that are being
                merged (the parameters specified by the tuple keys in
                output).

        Returns:
            object: Generated output.

        """
        if name == 'render':
            return np.concatenate(list(output.values()), axis=1)
        return super(RenderTask, self)._merge_output(
            name, output, merged_param)

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id != 'all'
        if name == 'render_camera':
            return self._generate_camera()
        elif name == 'render':
            if self.args.scene_age.age is not None:
                before = self.args.output_render.path
                camera_args = self.get_output('render_camera')
                for k, v in camera_args.items():
                    setattr(self.args, k, v)
                after = self.args.output_render.path
                assert after == before
            return self._render_scene(self.args.query)
        return super(RenderTask, self)._generate_output(name)


class TotalsTask(TemporalTaskBase):
    r"""Class for plotting the flux on a geometry as a function of time."""
    _name = 'totals'
    _step_task = RayTraceTask
    _output_info = {
        'totals': {
            'base_prefix': True,
            'ext': '.json',
            'description': 'raytraced query totals',
            'composite_param': RayTraceTask._output_info[
                'raytrace']['composite_param'],
        },
        'totals_plot': {
            'ext': '.png',
            'base_output': 'totals',
            'description': 'a plot of raytraced query totals',
            'merge_all': True,
            'merge_all_output': 'totals',
        },
    }
    _external_tasks = {
        RayTraceTask: {
            'exclude': [
                'show_rays', 'ray_color', 'ray_width', 'ray_length',
                'arrow_width', 'highlight', 'highlight_color',
                'traced_mesh_colormap', 'query',
            ],
            'modifications': {
                'canopy': {
                    'append_choices': ['all'],
                },
                'periodic_canopy': {
                    'append_choices': ['all'],
                },
            },
        },
    }
    _arguments = [
        (('--totals-query', ), {
            'type': str, 'choices': RayTraceTask._query_options_stats,
            'default': 'total_flux',
            'help': ('Name of the raytracer query result that should '
                     'be plot as a function of time. '),
            'suffix_param': {
                'cond': True,
                'skip_outputs': ['totals'],
                'outputs': ['totals_plot'],
            },
        }),
        (('--per-plant', ), {
            'action': 'store_true',
            'help': ('Plot the totals on a per-plant basis (valid for '
                     'totals_plot only'),
            'suffix_param': {
                'value': 'perPlant',
                'outputs': ['totals_plot'],
            },
        }),
    ]

    @classmethod
    def _read_output(cls, args, name, fname):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to read.
            fname (str): Path of file that should be read from.

        Returns:
            object: Contents of the output file.

        """
        out = super(TotalsTask, cls)._read_output(args, name, fname)
        if name == 'totals':
            out['times'] = [
                datetime.fromisoformat(x) for x in out['times']
            ]
        return out

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if name == 'totals':
            output = dict(output,
                          times=[x.isoformat() for x in output['times']])
        super(TotalsTask, cls)._write_output(args, name, fname, output)

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        if args.canopy == 'single':
            args.per_plant = False
        iterating = [
            k for k in cls._output_info['totals']['composite_param']
            if getattr(args, k) == 'all'
        ]
        if ((args.figure_color_by is None
             and args.totals_query.startswith('profile_'))):
            args.figure_color_by = 'time'
        if args.per_plant:
            if args.figure_color_by is None:
                if iterating and args.figure_linestyle_by is None:
                    args.figure_color_by = 'label'
                    if len(iterating) == 1:
                        args.figure_linestyle_by = iterating[0]
                    else:
                        args.figure_linestyle_by = 'label'
                else:
                    args.figure_color_by = 'location'
            elif args.figure_linestyle_by is None:
                args.figure_linestyle_by = 'location'
        if args.overwrite_totals:
            args.overwrite_totals_plot = True
        super(TotalsTask, cls).adjust_args_internal(args, **kwargs)

    @cached_property
    def time_marker(self):
        r"""matplotlib.lines.line2D: Vertical line marking a time."""
        return self.axes.axvline(x=self.args.start_time.time,
                                 color=(0, 0, 0),
                                 alpha=0.5, linewidth=10)

    def mark_time(self, time):
        r"""Mark a time with a vertical line on the figure.

        Args:
            time (datetime.datetime): Time that should be marked.

        """
        self.time_marker.set_xdata([time, time])

    def plot_query(self, query, x, y, label=None, plantid=None,
                   reset=False, line_prop=None):
        r"""Plot the data for a single crop ID.

        Args:
            query (str): Name of property contained by y.
            x (np.ndarray): X values.
            y (np.ndarray): Query values for each x value.
            label (str, optional): Label for lines.
            plantid (int, optional): Plant ID for provided data.
            reset (bool, optional): Reset the figure.
            line_prop (dict, optional): Parameters to pass to
                get_line_properties to determine line properties.

        """
        profile = query.startswith('profile_')
        if line_prop is None:
            line_prop = {}
        ax = self.axes
        first = (reset or not hasattr(self, '_lines'))
        if first:
            self._lines = {}
            self._linestyles = {}
            self._linecolors = {}
            ax.cla()
        if label is None:
            label = self.args.id
        self._lines.setdefault(label, {'interior': 0, 'exterior': 0})
        if first:
            if self.args.figure_font_weight:
                from matplotlib import rcParams
                rcParams['font.weight'] = self.args.figure_font_weight
            if profile:
                xlabel = 'Height'
            else:
                xlabel = 'Time'
            ylabel = query if query in ['LAI'] else query.title()

            def add_units_label(v):
                v_units = ''
                if isinstance(v, units.QuantityArray):
                    if v.units:
                        v_units = str(v.units)
                elif isinstance(v[0], units.Quantity):
                    if v[0].units:
                        v_units = str(v[0].units)
                if not v_units:
                    return ''
                return f" ({v_units})"

            xlabel += add_units_label(x)
            ylabel += add_units_label(y)
            ax.set_xlabel(xlabel, weight=self.args.figure_font_weight)
            ax.set_ylabel(ylabel.replace('_', ' '),
                          weight=self.args.figure_font_weight)
        layout_task = self.output_task('layout')
        locStr = None
        full_label = None
        if plantid is not None:
            if layout_task.isExteriorPlant(int(plantid)):
                locStr = 'exterior'
            else:
                locStr = 'interior'
            self._lines[label][locStr] += 1
            full_label = f'{locStr.title()} plants ({label})'
        else:
            full_label = label
        line_prop['label'] = full_label
        line_prop['location'] = locStr
        line_prop['class'] = label
        line_kws = self.get_line_properties(self.args, **line_prop)
        if plantid is None or self._lines[label][locStr] == 1:
            line_kws['label'] = full_label
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        # if isinstance(x, units.QuantityArray):
        #     idxsort = np.argsort(x.value)
        # else:
        #     idxsort = np.argsort(x)
        # x = x[idxsort]
        # y = y[idxsort]
        ax.plot(x, y, **line_kws)

    def plot_data(self, values, label=None, reset=False, tprofile=None,
                  **kwargs):
        r"""Plot the data for a single crop ID.

        Args:
            values (dict): Mapping between query name and dictionaries
                of values for each component.
            label (str, optional): Label that should be used for lines.
            reset (bool, optional): Reset the figure.
            **kwargs: Additional keyword arguments are passed to calls to
                plot_query.

        """
        if label is None:
            label = self.args.id
        times = values['times']
        query = self.args.totals_query
        if query in RayTraceTask._query_options_phy:
            query = f'total_{query}'
        profile = query.startswith('profile_')
        tidx = None
        if profile:
            cmap = None
            norm = None
            if self.args.figure_color_by == 'time':
                import matplotlib as mpl
                norm = mpl.colors.Normalize(
                    vmin=0,
                    vmax=(times[-1] - times[0]).total_seconds(),
                )
                cmap = mpl.colormaps['viridis']
                kwargs.setdefault('line_prop', {})
            if tprofile is None:
                for tprofile in times:
                    if cmap is not None:
                        color = cmap(norm(
                            (tprofile - times[0]).total_seconds()))
                        kwargs['line_prop']['color'] = color
                    self.plot_data(values, label=f'{label} {tprofile}',
                                   reset=reset, tprofile=tprofile,
                                   **kwargs)
                    reset = False
                return
            if isinstance(times, np.ndarray):
                tidx = np.where(times == tprofile)[0][0]
            else:
                tidx = times.index(tprofile)
            x = values[f'{query}_bins']
        else:
            x = times
        y = values[query]
        if query.startswith(tuple(
                RayTraceTask._query_options_calc_prefix_per_plant)):
            if self.args.per_plant:
                for i, iy in y.items():
                    if i == 'total':
                        continue
                    ix = x
                    if profile:
                        ix = x[i][tidx, :]
                        iy = iy[tidx, :]
                    self.plot_query(query, ix, iy,
                                    label=label, plantid=i, reset=reset,
                                    **kwargs)
                    reset = False
                return
            y = y['total']
            if profile:
                x = x['total']
        if profile:
            x = x[tidx, :]
            y = y[tidx, :]
        self.plot_query(query, x, y, label=label,
                        reset=reset, **kwargs)

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        value = x.get_output('raytrace_stats')
        return (x.args.time.time, value)

    def join_steps(self, xlist):
        r"""Join the output form all of the steps.

        Args:
            xlist (list): Result of all steps.

        Returns:
            object: Joined output from all steps.

        """
        assert self.args.id != 'all'
        times = [x[0] for x in xlist]
        values = [x[1] for x in xlist]
        out = RayTraceTask.join_query_stats(values, per_plant=True)
        out['times'] = times
        return out

    def _merge_output(self, name, output, merged_param):
        r"""Merge the output for multiple sets of parameters values.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from tuples of parameter values to
               the output for the parameter values.
            merged_param (tuple): Names of the parameters that are being
                merged (the parameters specified by the tuple keys in
                output).

        Returns:
            object: Generated output.

        """
        if name == 'totals_plot':
            for param, values in output.items():
                kws = {k: v for k, v in zip(merged_param, param)}
                labels = []
                for k, v in kws.items():
                    if k in ['id', 'canopy']:
                        labels.append(str(v))
                    else:
                        labels.append(f'{k} = {v}')
                self.plot_data(values, label=', '.join(labels),
                               line_prop=kws)
            self.axes.legend()
            return self.figure
        return super(TotalsTask, self)._merge_output(
            name, output, merged_param)

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name == 'totals_plot':
            assert self.args.id != 'all'
            out = self.get_output('totals')
            self.plot_data(out)
            self.axes.legend()
            return self.figure
        return super(TotalsTask, self)._generate_output(name)


class AnimateTask(TemporalTaskBase):
    r"""Class for producing an animation."""

    _name = 'animate'
    _step_task = RenderTask
    _step_args_preserve = ['render_colormap']
    _output_info = {
        'animate': {
            'base_output': 'render',
            'description': 'the animation',
        },
    }
    _external_tasks = {
        RenderTask: {
            'modifications': {
                'output_render': {
                    'default': True,
                },
            }
        },
        TotalsTask: {
            'optional': True,
            'increment_suffix_index': 1,  # After local arguments
            'modifications': {
                'totals_query': {
                    'default': None,
                    'append_suffix_param': {
                        'cond': False,
                    },
                },
            },
        },
    }
    _arguments = [
        (('--movie-format', ), {
            'type': str, 'choices': ['mp4', 'mpeg', 'gif'],
            'default': 'gif',
            'help': 'Format that the movie should be output in',
            'suffix_param': {'prefix': '.', 'cond': False},
        }),
        (('--frame-rate', ), {
            'type': int, 'default': 1,
            'help': ('The frame rate that should be used for the '
                     'generated movie in frames per second'),
            'suffix_param': {'suffix': 'fps', 'noteq': 1},
        }),
        (('--inset-totals', ), {
            'action': 'store_true',
            'help': ('Inset a plot of the query total below the '
                     'image.'),
            'suffix_param': {
                'value': lambda x: (
                    'totals' + (
                        x.totals_query.title().replace('_', '')
                        if x.totals_query.split('total_', 1)[-1] != x.query
                        else ''
                    )
                ),
            },
        }),
    ]

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        if args.totals_query is None:
            if args.query in RayTraceTask._query_options_phy:
                args.totals_query = f'total_{args.query}'
            else:
                args.totals_query = args.query
        super(AnimateTask, cls).adjust_args_internal(args, **kwargs)

    def __init__(self, *args, **kwargs):
        self._inset_figure = None
        super(AnimateTask, self).__init__(*args, **kwargs)

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        utils.write_movie(output, fname, frame_rate=args.frame_rate,
                          verbose=args.verbose)

    @classmethod
    def _output_ext(cls, args, name, wildcards=None, skipped=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skipped (list, optional): List of arguments that should be
                skipped in the generated output file name.

        Returns:
            str: Output file extension.

        """
        return cls._arguments['movie_format'].generate_suffix(
            args, name, wildcards=wildcards, skipped=skipped,
            force=True,
        )

    @property
    def totals_task(self):
        r"""TotalsTask: Totals instance."""
        return self.external_tasks['totals']

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        assert isinstance(x.output_file('render'), str)
        if not self.args.inset_totals:
            return x.output_file('render')
        old_frame = x.output_file('render')
        new_suffix = self._arguments['inset_totals'].generate_suffix(
            self.args, 'animate')
        new_frame = f'_{new_suffix}'.join(os.path.splitext(old_frame))
        old_data = x.get_output('render')
        if self._inset_figure is None:
            self._inset_figure = self.totals_task.generate_output(
                'totals_plot')
            width_px = old_data.shape[1]
            height_px = int(0.2 * width_px)
            dpi = self._inset_figure.get_dpi()
            figsize = ((width_px + 1) / dpi, (height_px + 1) / dpi)
            self._inset_figure.set_size_inches(*figsize)
        self.totals_task.mark_time(x.args.time.time)
        add_data = self.totals_task.raw_figure_data
        add_data = add_data[:, :old_data.shape[1], :]
        new_data = np.concatenate([old_data, add_data], axis=0)
        utils.write_png(new_data, new_frame, verbose=self.args.verbose)
        return new_frame


class MatchQuery(OptimizationTaskBase):
    r"""Class for matching raytrace query."""

    _step_task = RayTraceTask
    _name = 'match_query'
    _final_outputs = ['totals_plot']
    _arguments = [
        (('--goal', ), {
            'help': 'Goal of the optimization.',
            'default': 'scene_average_flux',
            'choices': RayTraceTask._query_options,
            'suffix_param': {},
        }),
        (('--vary', ), {
            'type': str,
            'help': 'Argument that should be varied.',
            'default': 'row_spacing',
            'suffix_param': {'prefix': 'vs_'},
        }),
        OptimizationTaskBase._arguments['method'],
        (('--goal-id', ), {
            'help': (
                'ID that the goal of the optimization should be matched '
                'to.'),
            'suffix_param': {
                'prefix': 'matchTo_',
                'index': -1,
            },
        }),
    ]

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        args.final_args = {}
        if args.canopy not in ['single', 'virtual', 'virtual_single']:
            # Regenerating unique canopies is very time intensive
            args.final_args['canopy'] = args.canopy
            args.canopy = 'virtual_single'
            if ((args.periodic_canopy == 'scene'
                 and args.nrows and args.ncols)):
                args.final_args['periodic_canopy'] = args.periodic_canopy
                args.periodic_canopy = False
        if args.goal_id is None:
            ids = GenerateTask.all_ids_class(args)
            assert ids
            for x in ids:
                if x != args.id:
                    args.goal_id = x
                    break
        super(MatchQuery, cls).adjust_args_internal(args, **kwargs)

    @cached_property
    def goal(self):
        r"""units.Quantity: Value that should be achieved."""
        assert self.args.id != self.args.goal_id
        args_overwrite = {
            'id': self.args.goal_id,
            self.args.vary: getattr(self.args, self.args.vary)
        }
        inst = self.run_iteration(
            cls=self._step_task,
            args_overwrite=args_overwrite,
        )
        out = self.finalize_step(inst, force_output=True)
        self.log(f'GOAL: {out}', force=True)
        return out

    def finalize_step(self, x, force_output=False):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.
            force_output (bool, optional): If True, force output to
                disk.

        Returns:
            object: Finalized step result.

        """
        if getattr(self, '_prev_instance', None) is None:
            force_output = True
        return x.calculate_query(
            self.args.goal, prevent_output=(not force_output)
        )

    def final_output_args(self, name):
        r"""Get the arguments that should be used generate the final
        output.

        Args:
            name (str): Name of the final output to generate.

        Returns:
            dict: Arguments to use.

        """
        out = copy.deepcopy(self.args.final_args)
        if name in ['totals_plot']:
            out['totals_query'] = self.args.goal
        return out

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name == 'totals_plot':
            goal_totals = super(MatchQuery, self)._generate_output(
                name, output_name='totals',
                args_overwrite=dict(
                    self.final_output_args(name),
                    id=self.args.goal_id,
                ),
            )
            out = super(MatchQuery, self)._generate_output(
                name, output_name='instance')
            match_totals = out.get_output('totals')
            out.plot_data(goal_totals, label=self.args.goal_id,
                          reset=True)
            out.plot_data(match_totals, label=self.args.id)
            out.axes.legend()
            return out.figure
        return super(MatchQuery, self)._generate_output(name)


__all__ = ["hothouse"]
