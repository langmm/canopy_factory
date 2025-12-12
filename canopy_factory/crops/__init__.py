import os
import copy
import numpy as np
import yggdrasil_rapidjson as rapidjson
from yggdrasil_rapidjson import units
from canopy_factory import utils
from canopy_factory.utils import (
    parse_units, parse_quantity, parse_axis, parse_color,
    get_class_registry, UnitSet, NoDefault,
    cached_property, readonly_cached_property,
)
from canopy_factory.cli import (
    TaskBase, TimeArgument, OutputArgument, SuffixGenerator,
)
from canopy_factory.crops import (
    monocot, maize,
    dicot, tomato,
)


############################################################
# TASKS
############################################################

class ParametrizeCropTask(TaskBase):
    r"""Class for generating the LSystem parameters for a canopy."""

    _name = 'parametrize'
    _help = 'Generate the parameters for an LSystem crop model.'
    _runtime_param = [
        'verbose', 'debug', 'debug_param', 'debug_param_prefix',
    ]
    _output_info = {
        'parametrize': {
            'base_string': '',
            'directory': utils.cfg['directories']['param'],
            'ext': '.json',
            'description': (
                'parameters used to generate 3D '
                'representations of a canopy'
            ),
            'composite_param': ['id', 'data_year'],
        },
        'lpy_model': {
            'base_string': '',
            'directory': utils.cfg['directories']['lpy'],
            'ext': '.lpy',
            'description': 'LPy L-system rules',
        },
    }
    _subparser_arguments = {
        'crop': ['id', 'data', 'data_year'],  # True to include params
    }
    _subparser_modifications = {
        'crop': {
            'id': {
                'append_choices': ['all'],
            },
            'data_year': {
                'append_choices': ['all'],
            },
        }
    }
    _arguments = [
        (('--debug-param', ), {
            'action': 'append',
            'help': ('Parameter(s) that debug mode should be enabled '
                     'for (does not enable debugging for children of '
                     'the named parameter(s). Use --debug-param-prefix '
                     'to also enable debugging for children)'),
        }),
        (('--debug-param-prefix', ), {
            'action': 'append',
            'help': ('Prefix(es) of parameters that debug mode should '
                     'be enabled for.'),
        }),
    ]
    _output_suffix = SuffixGenerator([
        ('crop', {'cond': True, 'prefix': ''}),
        ('id', {'cond': True, 'outputs': ['parametrize']}),
        ('data_year', {'outputs': ['parametrize']}),
    ])
    _age_strings = [
        'planting', 'maturity', 'senesce', 'remove',
    ]

    @staticmethod
    def _on_registration(cls):
        units_args = {
            'output': {
                'defaults': {
                    'length': 'cm',
                    'mass': 'kg',
                    'time': 'days',
                    'angle': 'degrees',
                },
                'help_template': (
                    'Units that {plural} should be converted to in '
                    'the generated parameter file'
                ),
            },
            'data': {
                'defaults': {
                    'length': 'cm',
                    'mass': 'kg',
                    'time': 'days',
                    'angle': 'degrees',
                },
                'help_template': (
                    'Units that {plural} are in within the provided '
                    'data file'
                )
            },
        }
        for k, v in units_args.items():
            UnitSet.add_unit_arguments(cls, k, **v)
        TaskBase._on_registration(cls)

    def __init__(self, *args, **kwargs):
        super(ParametrizeCropTask, self).__init__(*args, **kwargs)
        if ((isinstance(self.args.id, str)
             and self.args.id.startswith('all') and not self.is_root)):
            self.ensure_initialized()

    def ensure_initialized(self):
        r"""Initialize the output for another task."""
        if self.is_root:
            return
        if isinstance(getattr(self.args, f'output_{self._name}', None),
                      OutputArgument):
            return
        if ((isinstance(self.args.id, str)
             and self.args.id.startswith('all'))):
            self.args = copy.deepcopy(self.args)
            self.args.id = utils.DataProcessor.base_id_from_file(
                self.args.data, crop=self.args.crop,
                year=self.args.data_year,
            )
        self.finalize()

    @classmethod
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        super(ParametrizeCropTask, cls).adjust_args_internal(args)
        if args.data:
            args.data_year = utils.DataProcessor.from_file(args.data).year
        elif args.data_year:
            args.data = utils.DataProcessor.output_name(args.crop,
                                                        args.data_year)
            if not os.path.isfile(args.data):
                args.data = None
                args.data_year = None

    @cached_property
    def generator_class(self):
        r"""type: Plant generator class that should be parameterized."""
        return get_class_registry().get('crop', self.args.crop)

    @cached_property
    def generator(self):
        r"""PlantGenerator: Parametrized plant generator."""
        param = dict(self.parameters, **self.runtime_parameters)
        return self.generator_class(**param)

    @cached_property
    def runtime_parameters(self):
        r"""dict: Set of runtime parameters controling log."""
        return {k: getattr(self.args, k) for k in self._runtime_param}

    @cached_property
    def parameters(self):
        r"""dict: Set of model parameters collected from the command line."""
        return self.get_output('parametrize')

    @classmethod
    def get_age_from_parameters(cls, name, parameters):
        r"""Get the age at which the generated crop is mature.

        Args:
            name (str): Age that should be returned.
            parameters (dict): Parameters that should be used.

        Returns:
            units.Quantity: Age of maturity.

        """
        if name == 'planting':
            return units.Quantity(0.0, 'days')
        elif name == 'maturity':
            return (
                parameters['NMax']
                * parameters['Plastocron']
            )
        elif f'Age{name.title()}' in parameters:
            return parameters[f'Age{name.title()}']
        else:
            raise NotImplementedError(f'Invalid age \"{name}\"')

    def get_age(self, name):
        r"""Get the age at which the generated crop is mature.

        Args:
            name (str): Age that should be returned.

        Returns:
            units.Quantity: Age of maturity.

        """
        return self.get_age_from_parameters(name, self.parameters)

    @classmethod
    def get_age_class(cls, args, name):
        r"""Get the age at which the generated crop is mature.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Age that should be returned.

        Returns:
            units.Quantity: Age of maturity.

        """
        param = cls.from_external_args(args)
        return param.get_age(name)

    def generate_parameters(self):
        r"""Generate the model parameters.

        Returns:
            dict: Architecture parameters.

        """
        kwargs = {}
        for k, v in self.generator_class._arguments:
            kattr = self.arg2dest(k, v)
            if getattr(self.args, kattr, None) is not None:
                kwargs[kattr] = getattr(self.args, kattr)
        for k in self._runtime_param:
            kwargs[k] = getattr(self.args, k)
        inst = self.generator_class(**kwargs)
        out = copy.deepcopy(inst.all_parameters)
        if self.args.data:
            self.generator_class.parameters_from_file(self.args, out)
            out['data'] = os.path.basename(out['data'])
        if 'AgeRemove' not in out:
            out['AgeRemove'] = (
                out['NMax'] * out['Plastocron']
            )
        output_units = UnitSet.from_attr(
            self.args, prefix='output_units_')
        for k in list(out.keys()):
            if isinstance(out[k], units.QuantityArray):
                out[k] = output_units.convert(out[k])
        out.update(
            output_units.as_dict(suffix='_units', as_strings=True)
        )
        return out

    @cached_property
    def defaults(self):
        r"""dict: Set of default model parameters."""
        out = {
            'RUNTIME_PARAM': {},
            'OUTPUT_TIME': 20,
        }
        return out

    def add_parameters_to_lpy_model(self):
        r"""Add parameters to the lpy_model."""
        comment_start = '\n'.join([
            60 * '#',
            "## WARNING: THE FOLLOWING SECTION IS GENERATED AND SHOULD ",
            "##   NOT BE MODIFIED DIRECTLY",
            60 * '#',
        ]) + '\n'
        comment_end = '\n' + '\n'.join([
            60 * '#',
            "## WARNING: THE PREVIOUS SECTION IS GENERATED AND SHOULD ",
            "##   NOT BE MODIFIED DIRECTLY",
            60 * '#',
        ])
        with open(self.output_file('lpy_model'), 'r') as fd:
            contents = fd.read()
        prefix_contents = ''
        suffix_contents = ''
        if comment_start in contents:
            assert comment_end in contents
            prefix_contents = contents.split(comment_start)[0]
            if prefix_contents.endswith('\n'):
                prefix_contents = prefix_contents[:-1]
            suffix_contents = contents.split(comment_end)[-1]
            if suffix_contents.startswith('\n'):
                suffix_contents = suffix_contents[1:]
        contents = [
            f'extern({k} = {rapidjson.dumps(v)})'
            for k, v in self.defaults.items()
        ] + [
            'from canopy_factory.utils import get_class_registry',
            f'generator_cls = get_class_registry().get("crop", '
            f'"{self.args.crop}")',
            'generator = generator_cls(context=context(), **RUNTIME_PARAM)'
        ]
        contents = [
            prefix_contents, comment_start
        ] + contents + [
            comment_end, suffix_contents
        ]
        with open(self.output_file('lpy_model'), 'w') as fd:
            fd.write('\n'.join(contents))

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name == 'lpy_model':
            return self.generator.lsystem
        elif name == 'parametrize':
            return self.generate_parameters()
        super(ParametrizeCropTask, self)._generate_output(name)


class LayoutTask(TaskBase):
    r"""Class for plotting the layout of a canopy."""

    _name = 'layout'
    _output_info = {
        'layout': {
            'ext': '.png',
            'description': 'a plot of the layout',
        },
    }
    _composite_arguments = {
        'time': {
            'time': {
                'description': ' that the sun should be modeled for',
                'ignore': ['age', 'planting_date'],
                'optional': True,
            },
        }
    }
    _argument_conversions = {
        'mesh_units': [
            'plot_length', 'plot_width', 'row_spacing', 'plant_spacing',
            'x', 'y',
            'ground_height',
        ],
    }
    _arguments = [
        (('--canopy', ), {
            'choices': ['single', 'tile', 'unique'],
            'default': 'unique',
            'help': 'Type of canopy to layout',
        }),
        (('--plot-length', '--row-length'), {
            'type': parse_quantity, 'default': 200, 'units': 'cm',
            'help': 'Length of plot rows forming canopy (in cm)',
        }),
        (('--plot-width', ), {
            'type': parse_quantity, 'units': 'cm',
            'help': ('Width of plot forming canopy (in cm). If provided '
                     '\'nrows\' will be determined based on the provided '
                     '\'row_spacing\'. If not provided, \'plot_width\' '
                     'will be determined from \'nrows\' and '
                     '\'row_spacing\'.'),
        }),
        (('--nrows', ), {
            'type': int, 'default': 4,
            'help': 'Number of rows to generate in plot',
        }),
        (('--ncols', ), {
            'type': int,
            'help': 'Number of plants to generate in each row',
        }),
        # (('--match-id', ), {
        #     'type': str, 'nargs': '?', 'const': True,
        #     'help': ('Match the \"--match-goal\" raytraced value '
        #              'for this ID by varying the \"--match-vary\" '
        #              'argument.'),
        # }),
        # (('--match-goal', ), {
        #     'type': str, 'default': 'flux_per_area',
        #     'help': ('Raytraced property that should be matched to '
        #              '\"--match-id\" by varying \"--match-vary\" '
        #              'argument.'),
        # }),
        # (('--match-vary', ), {
        #     'type': str, 'default': 'row_spacing',
        #     'help': ('Argument that should be varied in order to match '
        #              'the raytraced property from \"--match-goal\" to '
        #              '\"--match-id\".'),
        # }),
        (('--row-spacing', ), {
            'type': parse_quantity, 'default': 76.2, 'units': 'cm',
            'help': 'Space between adjacent rows in plot (in cm)',
        }),
        (('--plant-spacing', '--col-spacing'), {
            'type': parse_quantity, 'default': 18.3, 'units': 'cm',
            'help': 'Space between adjacent plants in rows (in cm)',
        }),
        (('-x', '--x', '--row-offset'), {
            'type': parse_quantity, 'default': 0.0, 'units': 'cm',
            'help': ('Starting position in the x direction '
                     '(perpendicular to rows)'),
        }),
        (('-y', '--y', '--plant-offset'), {
            'type': parse_quantity, 'default': 0.0, 'units': 'cm',
            'help': ('Starting position in the y direction (along '
                     'rows)'),
        }),
        (('--plantid', ), {
            'type': int, 'default': 0,
            'help': 'Starting plant ID',
        }),
        (('--periodic-canopy', ), {
            'nargs': '?', 'const': 'scene', 'default': False,
            'choices': [False, 'scene', 'rays'],
            'help': ('Make the canopy periodic for ray tracing so '
                     'that is infinitely wide')
        }),
        (('--periodic-canopy-count', ), {
            'type': int,
            'help': ('Number of times the canopy should be repeated in '
                     'each direction'),
        }),
        (('--axis-up', ), {
            'type': parse_axis, 'default': 'y',
            'help': 'Axis along which plants should grow within the mesh',
        }),
        (('--axis-rows', ), {
            'type': parse_axis, 'default': 'z',
            'help': 'Axis along which rows should be spaced',
        }),
        (('--axis-north', ), {
            'type': parse_axis, 'default': 'x',
            'help': ('Axis that should represent north when computing '
                     'incident solar radiation'),
        }),
        (('--ground-height', ), {
            'type': parse_quantity, 'default': 0.0, 'units': 'meters',
            'help': ('Distance that the ground is above 0 along the '
                     '\"axis_up\" direction'),
        }),
        # TODO: Use mesh units in input
        (('--mesh-units', ), {
            'type': parse_units, 'default': units.Units('cm'),
            'help': 'Units that mesh should be output in',
        }),
        (('--interior-plant-color', ), {
            'type': parse_color, 'default': 'blue',
            'help': ('Color that should be used for interior plants. '
                     'This should be a color string or 3 '
                     'comma separated RGB values expressed as floats in '
                     'the range [0, 1]'),
        }),
        (('--exterior-plant-color', ), {
            'type': parse_color, 'default': 'green',
            'help': ('Color that should be used for exterior plants. '
                     'This should be a color string or 3 '
                     'comma separated RGB values expressed as floats in '
                     'the range [0, 1]'),
        }),
        (('--periodic-plant-color', ), {
            'type': parse_color, 'default': 'grey',
            'help': ('Color that should be used for buffer plants added '
                     'to periodic canopies. This should be a color '
                     'string or 3 comma separated RGB values expressed '
                     'as floats in the range [0, 1]'),
        }),
    ]
    _output_suffix = SuffixGenerator([
        ('canopy', {
            'prefix': '_canopy', 'title': True, 'noteq': 'single',
        }),
        ('periodic', SuffixGenerator(
            [
                ('periodic_canopy_count', {
                    'value': lambda x: getattr(
                        x, 'periodic_canopy_count_array', NoDefault),
                    'sep': 'x',
                }),
                ('periodic_canopy', {}),
            ],
            cond=lambda x: bool(x.periodic_canopy),
        )),
        ('geometry', SuffixGenerator(
            [
                (('row_spacing', 'plant_spacing'), {
                    'kwargs_set': {'sep': 'x'},
                    'noteq': {
                        'row_spacing': parse_quantity(76.2, 'cm'),
                        'plant_spacing': parse_quantity(18.3, 'cm'),
                    },
                }),
                (('nrows', 'ncols'), {
                    'kwargs_set': {'sep': 'x'},
                    'noteq': {
                        'nrows': (
                            lambda x: (1 if x.canopy == 'single'
                                       else 4)
                        ),
                        'ncols': (
                            lambda x: (1 if x.canopy == 'single'
                                       else 10)
                        ),
                    },
                }),
            ],
            cond=lambda x: (x.periodic_canopy or x.canopy != 'single'),
        )),
        ('time', {'composite': 'time'}),
    ])

    @classmethod
    def _convert_mesh_units(cls, args, k, v):
        if isinstance(args.mesh_units, str):
            args.mesh_units = units.Units(args.mesh_units)
        setattr(args, k, parse_quantity(v, args.mesh_units))

    @classmethod
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        super(LayoutTask, cls).adjust_args_internal(args)
        if args.canopy == 'single':
            args.nrows = 1
            args.ncols = 1
            args.plot_width = args.row_spacing
            args.plot_length = args.plant_spacing
        else:
            if args.periodic_canopy_count is None:
                args.periodic_canopy_count = 2
        if args.plot_width is None:
            args.plot_width = args.nrows * args.row_spacing
        if args.plot_length is None:
            args.plot_length = args.ncols * args.plant_spacing
        args.nrows = int(args.plot_width / args.row_spacing)
        args.ncols = int(args.plot_length / args.plant_spacing)
        args.axis_cols = np.cross(args.axis_up, args.axis_rows)
        args.axis_east = np.cross(args.axis_north, args.axis_up)
        if args.periodic_canopy is True:
            args.periodic_canopy = 'scene'
        if args.periodic_canopy:
            args.periodic_period = np.array([
                args.nrows * args.row_spacing,
                args.ncols * args.plant_spacing,
                0.0
            ], 'f4')
            args.periodic_direction = np.vstack([
                args.axis_rows,
                args.axis_cols,
                args.axis_up,
            ])
            if args.periodic_canopy_count is None:
                assert args.canopy == 'single'
                periodic_canopy_count = 2
                nrows = 4
                ncols = 10
                args.periodic_canopy_count_array = np.array([
                    periodic_canopy_count * nrows,
                    periodic_canopy_count * ncols,
                    0,
                ], 'i4')
            else:
                args.periodic_canopy_count_array = np.array([
                    args.periodic_canopy_count,
                    args.periodic_canopy_count,
                    0,
                ], 'i4')

    @cached_property
    def nplants(self):
        r"""int: Total number of plants in the canopy."""
        # TODO: Allow all?
        return self.args.nrows * self.args.ncols

    @cached_property
    def plant_positions(self):
        r"""np.ndarray: Locations of each plant in the canopy, in plantid
        order."""
        # TODO: Allow multiple crop classes?
        pos0 = self.args.plant_spacing * np.zeros((1, 3), 'f4')
        plantid = 0
        x = self.args.x
        y = self.args.y
        axis_x = self.args.axis_rows
        axis_y = self.args.axis_cols
        out = self.args.plant_spacing * np.zeros((self.nplants, 3), 'f4')
        for i in range(self.args.nrows):
            y = self.args.y
            for j in range(self.args.ncols):
                out[plantid, :] = pos0 + x * axis_x + y * axis_y
                y = y + self.args.plant_spacing
                plantid += 1
            x = x + self.args.row_spacing
        return out

    @cached_property
    def nplants_periodic(self):
        r"""int: Number of plants in the periodic canopy buffer."""
        # TODO: Allow multiple crop classes?
        if not self.args.periodic_canopy:
            return 0
        out = np.prod(2 * self.args.periodic_canopy_count_array + 1) - 1
        if self.args.canopy != 'single':
            out *= self.nplants
        return out

    @cached_property
    def plant_positions_periodic(self):
        r"""np.ndarray: Locations of each plant in the periodic canopy
        buffer, in the order they are added to the scene."""
        # TODO: Allow all?
        if not self.args.periodic_canopy:
            return self.args.plant_spacing * np.zeros((0, 3), 'f4')
        from hothouse.scene import PeriodicScene as Scene
        pos_units0 = self.plant_positions.units
        pos0 = self.plant_positions
        shifts = units.QuantityArray(
            Scene.get_periodic_shifts(
                self.args.periodic_period.astype('f4'),
                self.args.periodic_direction.astype('f4'),
                self.args.periodic_canopy_count_array,
            ),
            pos_units0,
        )
        out = []
        for pos in pos0:
            out.append(shifts + pos)
        out = np.vstack(out)
        assert out.shape[0] == self.nplants_periodic
        assert out.shape[1] == 3
        return units.QuantityArray(out, pos_units0)

    def get_solar_model(self, time=None):
        r"""Get a solar model for the specified time.

        Args:
            time (str, datetime, optional): Time to get a solar model
                for. If not provided, the instance solar model will be
                returned.

        Returns:
            utils.SolarModel: Model for sun at provided time.

        """
        if time is None:
            return self.solar_model
        return TimeArgument.parse(time, self.args).solar_model

    def get_solar_direction(self, time=None):
        r"""Get the relative direction to the sun for the specified time.

        Args:
            time (str, datetime, optional): Time to get the solar
                direction for. If not provided, the instance solar
                direction will be returned.

        Returns:
            np.ndarray: Unit vector from the scene to the sun at the
                provided time.

        """
        if time is None:
            return self.solar_direction
        return self.get_solar_model(time).relative_direction(
            self.args.axis_up, self.args.axis_north)

    def isExteriorPlant(self, plantid, nbuffer_col=1, nbuffer_row=1):
        r"""Determine if a plant is on the edge of the field.

        Args:
            plantid (int): Plant identifier.
            nbuffer_col (int, optional): Number of plants from the edge
                along the columns to count as exterior.
            nbuffer_row (int, optional): Number of plants from the edge
                along the rowss to count as exterior.

        Returns:
            bool: True if the plant is on the edge, False otherwise.

        """
        plantid -= self.args.plantid
        j = plantid % self.args.ncols
        i = np.floor(plantid / self.args.ncols)
        return ((j < nbuffer_row)
                or (i < nbuffer_col)
                or (j >= (self.args.ncols - nbuffer_row))
                or (i >= (self.args.nrows - nbuffer_col)))

    @cached_property
    def solar_model(self):
        r"""utils.SolarModel: Model for sun."""
        return self.args.time.solar_model

    @cached_property
    def solar_direction(self):
        r"""np.ndarray: Unit vector from the scene to the sun."""
        if self.solar_model is None:
            return None
        return self.solar_model.relative_direction(
            self.args.axis_up, self.args.axis_north)

    @cached_property
    def solar_elevation(self):
        r"""units.Quantity: Apparent elevation of the sun."""
        if self.solar_model is None:
            return None
        return self.solar_model.apparent_elevation

    @cached_property
    def scene_layout(self):
        r"""dict: Parameters describing the scene layout."""
        out = {
            'plants': self.project_onto_ground(self.plant_positions),
            'periodic_plants': self.project_onto_ground(
                self.plant_positions_periodic),
            'north': self.project_onto_ground(
                self.args.axis_north, ray=True),
            'east': self.project_onto_ground(
                self.args.axis_east, ray=True),
        }
        return out

    @cached_property
    def subplot_ratio(self):
        r"""double: Ratio of subplot width to height."""
        width = (self.args.plot_width / self.args.row_spacing) + 10
        return width

    @cached_property
    def subplots(self):
        r"""Matplotlib subplots."""
        figbuf = 0.05
        return self.figure.subplots(
            1, 2,  # width_ratios=[1, self.subplot_ratio],
            gridspec_kw={
                'width_ratios': [1, self.subplot_ratio],
                'hspace': 0.0, 'wspace': figbuf / 2,
                'left': figbuf, 'right': 1.0 - figbuf,
                'bottom': figbuf, 'top': 1.0 - figbuf,
            },
        )

    @cached_property
    def axes(self):
        r"""Matplotlib axes."""
        if self._name != 'layout':
            return super(LayoutTask, self).axes
        out = self.subplots[1]
        out.set_axis_off()
        out.axis('equal')
        if self.args.plot_width > self.args.plot_length:
            out.autoscale(enable=True, axis='x', tight=True)
        else:
            out.autoscale(enable=True, axis='y', tight=True)
        return out

    @cached_property
    def legend_axes(self):
        r"""Matplotlib axes."""
        if self._name != 'layout':
            return None
        out = self.subplots[0]
        out.set_axis_off()
        out.axis('equal')
        return out

    def _get_color(self, plantid):
        if self.isExteriorPlant(plantid):
            return self.args.exterior_plant_color
        return self.args.interior_plant_color

    def project_onto_ground(self, pos, ray=False):
        r"""Project a 3D point onto the ground.

        Args:
            pos (np.ndarray): Set of one or more 3D positions.
            ray (bool, optional): If True, treat pos as a ray and
                normalize the returned projection.

        Returns:
            np.ndarray: x & y components of pos projected onto the
                scene ground.

        """
        pos_units = None
        if isinstance(pos, (units.Quantity, units.QuantityArray)):
            pos_units = pos.units
            pos = pos.data
        x = np.dot(pos, self.args.axis_rows)
        y = np.dot(pos, self.args.axis_cols)
        out = np.vstack([x, y]).T
        if pos.ndim == 1:
            out = out[0]
        if ray:
            out /= np.linalg.norm(out)
        elif pos_units is not None:
            out = units.QuantityArray(out, pos_units)
        return out

    def plot_sun(self, time, plant_min, plant_max,
                 arrow_length=0.1, nrays=9):
        r"""Plot the location of the sun for a given time as a set of
        rays at the edge of the field.

        Args:
            time (str, datetime): Time that the sun should be plotted
                for.
            plant_min (np.ndarray): Minimum extent of plant data in each
                dimension.
            plant_max (np.ndarray): Maximum extent of plant data in each
                dimension.
            arrow_length (float, optional): Length of rays relative to
                the field width.
            nrays (int, optional): Number of rays to plot.

        """
        plant_pad = np.array([
            self.args.row_spacing, self.args.row_spacing
        ]) / 5
        plant_min = plant_min - plant_pad
        plant_max = plant_max + plant_pad
        plant_mid = (plant_min + plant_max) / 2
        raysun = self.project_onto_ground(
            -self.get_solar_direction(time=time), ray=True)
        raysun *= arrow_length * (plant_max[0] - plant_min[0])
        rotsun = np.array([-raysun[1], raysun[0]])
        anglesun = np.arcsin(raysun[1] / np.linalg.norm(raysun))
        if raysun[0] < 0:
            anglesun = np.pi - anglesun
        anglesun = (2 * np.pi + anglesun) % (2 * np.pi)
        xsign = np.sign(-raysun[0])
        ysign = np.sign(-raysun[1])
        half = (plant_max - plant_min) / 2
        y = ysign * half[1]
        x = xsign * abs(y / np.tan(anglesun))
        if abs(x) > half[0]:
            x = xsign * half[0]
            y = ysign * abs(x * np.tan(anglesun))
        assert abs(x) <= half[0]
        assert abs(y) <= half[1]
        rayspacing = 1.0 / (nrays - 1)
        sunpos = np.array([plant_mid[0] + x, plant_mid[1] + y])
        for i in range(nrays):
            self.axes.arrow(
                *(sunpos - 0.5 * rotsun + i * rayspacing * rotsun),
                *raysun,
                color='orange',
            )
        if isinstance(time, str):
            textangle = (anglesun + (np.pi / 2)) % (2 * np.pi)
            ha = 'center'
            va = 'bottom'
            if textangle > np.pi / 2 and textangle < (3 * np.pi / 2):
                textangle -= np.pi
                va = 'top'
            self.axes.text(
                *sunpos, time,
                rotation=(180 * textangle / np.pi),
                rotation_mode='anchor',
                transform_rotates_text=True,
                horizontalalignment=ha,
                verticalalignment=va,
            )

    def plot_layout(self, layout):
        r"""Plot the layout.

        Args:
            layout (dict): Layout parameters.

        """
        ax = self.axes
        plantid = self.args.plantid
        for pos in layout['plants']:
            ax.plot(pos[0], pos[1],
                    color=self._get_color(plantid),
                    marker='o')
            ax.annotate(str(plantid), xy=pos, textcoords='data')
            plantid += 1
        for pos in layout['periodic_plants']:
            ax.plot(pos[0], pos[1],
                    color=self.args.periodic_plant_color,
                    marker='o')
        plant_min = np.vstack([layout['plants'].data,
                               layout['periodic_plants'].data]).min(axis=0)
        plant_max = np.vstack([layout['plants'].data,
                               layout['periodic_plants'].data]).max(axis=0)
        arrow_length = 0.1
        # Cardinal directions
        raynorth = arrow_length * layout['north']
        rayeast = arrow_length * layout['east']
        if rayeast[1] == 0:
            angle = 0
        else:
            angle = 180 * np.arcsin(layout['east'][1]) / np.pi
        dirpos = np.array([0.0, 0.2]) - 0.5 * raynorth - 0.5 * rayeast
        txtnorth = dirpos + raynorth - 0.05 * rayeast
        txteast = dirpos + rayeast
        self.legend_axes.arrow(*dirpos, *raynorth)
        self.legend_axes.arrow(*dirpos, *rayeast)
        self.legend_axes.text(
            *txtnorth, 'N',
            rotation=angle, rotation_mode='anchor',
            transform_rotates_text=True,
            horizontalalignment='right',
            verticalalignment='top',
        )
        self.legend_axes.text(
            *txteast, 'E',
            rotation=angle, rotation_mode='anchor',
            transform_rotates_text=True,
            horizontalalignment='right',
            verticalalignment='bottom',
        )
        # Sun direction
        if self.args.time.time:
            self.plot_sun(self.args.time.time, plant_min, plant_max,
                          arrow_length=arrow_length)
        else:
            for t in ['sunrise', 'sunset', 'noon']:
                self.plot_sun(t, plant_min, plant_max,
                              arrow_length=arrow_length)

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name == 'layout':
            self.plot_layout(self.scene_layout)
            return self.figure
        return super(LayoutTask, self)._generate_output(name)


class GenerateTask(TaskBase):
    r"""Class for generating 3D canopies."""

    _name = 'generate'
    _help = 'Generate a canopy mesh'
    _output_info = {
        'generate': {
            'base': 'parametrize',
            'description': 'mesh',
            'upstream': ['parametrize', 'lpy_model'],
            'merge_all': 'all_combined',
        },
        'geometryids': {
            'ext': '.csv',
            'base': 'generate',
            'description': (
                'the IDs of the plant/component each face '
                'belongs to'
            ),
            'merge_all': 'all_combined',
        },
    }
    _external_tasks = {
        LayoutTask: {
            'include': [
                'canopy', 'plot_length', 'plot_width',
                'nrows', 'ncols', 'row_spacing', 'plant_spacing',
                'x', 'y', 'plantid', 'periodic_canopy',
                'periodic_canopy_count', 'mesh_units',
            ],
            'modifications': {
                'canopy': {'default': 'single'},
            },
            'optional': True,
        },
        ParametrizeCropTask: {
            'include': [
                'crop', 'id', 'data', 'data_year',
                'debug_param', 'debug_param_prefix',
            ],
            'modifications': {
                'id': {
                    'append_choices': ['all', 'all_combined'],
                },
                'data_year': {
                    'append_choices': ['all', 'all_combined'],
                },
            },
        },
    }
    _composite_arguments = {
        'age': {
            'age': {
                'description': ' to generate model for',
                'ignore': ['planting_date'],
            },
        },
        'color': {
            'color': {
                'description': (
                    'that should be used for the generated plant. If '
                    'a value of \'plantid\' is provided, colors will '
                    'be used to identify individual plants by setting '
                    'the blue channel to the plant ID.'
                ),
                'optional': True,
            },
        },
    }
    _arguments = [
        # (('--age', ), {
        #     'type': parse_quantity, 'units': 'days',
        #     'help': ('Plant age to generate model for (in days '
        #              'since planting). Defaults to one day before the '
        #              'age of senescence.'),
        # }),
        # (('--derivation-length', ), {
        #     'type': int,
        #     'help': ('Number of iterations that should be produced '
        #              'for the L-system. If not provided, the derivation '
        #              'length will be set to allow for the maximum '
        #              'number of nodes to be achieved.'),
        # }),
        (('--location-stddev', ), {
            'type': float, 'default': 0.2,
            'help': ('Standard deviation relative to \'plant_spacing\' '
                     'that should be used when selecting planting '
                     'locations for multi-plant canopies'),
        }),
        (('--mesh-format', ), {
            'type': str, 'choices': utils._supported_3d_formats,
            'help': 'Format that mesh should be saved in',
        }),
        (('--verbose-lpy', ), {
            'action': 'store_true',
            'help': 'Enable debug messages within the lpy model',
        }),
        # (('--animate', ), {
        #     'action': 'store_true',
        #     'help': 'Generate a geometry for each iteration',
        # }),
    ]
    _output_suffix = SuffixGenerator([
        ('geometryids', SuffixGenerator([
            ('output', {'outputs': ['geometryids']}),
        ])),
        ('generate', SuffixGenerator(
            [
                (k, v) for k, v in
                LayoutTask._output_suffix.arguments.items()
                if k not in ['periodic', 'time']
            ] + [
                ('age', {'noteq': 'maturity'}),
                ('color', {'noteq': None}),
                ('plantid', {'cond': lambda x: (x.plantid > 0)}),
            ],
            outputs=['generate'],
        )),
    ])

    @classmethod
    def mesh_generated(cls, args):
        r"""Inspect the provided arguments to determine if the mesh is
        generated.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the mesh is generated, False otherwise.

        """
        if isinstance(args.output_generate, OutputArgument):
            return args.output_generate.generated
        return (not isinstance(args.output_generate, str))

    @classmethod
    def all_ids_class(cls, args):
        r"""Determine the complete set of IDs for the provided args.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            list: All crop classes for current arguments.

        """
        if not (cls.mesh_generated(args) and args.data):
            return []
        return utils.DataProcessor.from_file(args.data).ids

    @classmethod
    def base_id_class(cls, args):
        r"""Determine the base ID for the provided args.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            str: Base ID.

        """
        ids = cls.all_ids_class(args)
        if not ids:
            return None
        return ids[0]

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
        if ((args.age.string != 'maturity'
             and args.age.string not in fname)):
            print(args.age.string, fname)
            import pdb
            pdb.set_trace()
            raise AssertionError('Reading the wrong file')
        out = super(GenerateTask, cls)._read_output(args, name, fname)
        if name == 'geometryids':
            with open(fname, 'r') as fd:
                components = rapidjson.loads(fd.readline().lstrip('#'))
            return components, out
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
        if args.id == 'all':
            return
        if name == 'geometryids':
            return utils.write_csv(output[1], fname,
                                   comments=[rapidjson.dumps(output[0])])
        super(GenerateTask, cls)._write_output(args, name, fname, output)

    @classmethod
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.save_all_plantids = False
        if args.verbose:
            args.verbose_lpy = True
        if args.plantid > 0 and not args.save_all_plantids:
            args.dont_write_generate = True
            args.dont_write_geometryids = True
        super(GenerateTask, cls).adjust_args_internal(args)

    @classmethod
    def _output_ext(cls, args, name, wildcards=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Output file extension.

        """
        if name == 'generate':
            if not args.mesh_format:
                args.mesh_format = 'obj'
            return utils._inv_geom_ext[args.mesh_format]
        return None

    @cached_property
    def generator_class(self):
        r"""type: Plant generator class that should be parameterized."""
        return get_class_registry().get('crop', self.args.crop)

    @cached_property
    def generator(self):
        r"""PlantGenerator: Parametrized plant generator."""
        param = dict(self.parameters, **self.runtime_parameters)
        return self.generator_class(**param)

    def get_parameters(self):
        r"""Load/generate the parameter set for the model.

        Returns:
            dict: Loaded/generated parameter set.

        """
        if getattr(self.args, 'parameters', None) is None:
            self.args.parameters = self.get_output('parametrize')
        return self.args.parameters

    @cached_property
    def runtime_parameters(self):
        r"""dict: Set of runtime parameters controling log."""
        return {k: getattr(self.args, k) for k in
                ParametrizeCropTask._runtime_param}

    @cached_property
    def parameters(self):
        r"""dict: Runtime parameters for the lpy model."""
        return self.get_parameters()

    @cached_property
    def parameter_unit_system(self):
        r"""UnitSet: Unit system used by parameters."""
        return UnitSet.from_kwargs(self.parameters, suffix='_units')

    @readonly_cached_property
    def all_ids(self):
        r"""list: All crop classes for current data."""
        return self.generator_class.ids_from_file(self.args.data)

    @classmethod
    def shift_mesh(cls, args, mesh, x, y, plantid=None):
        r"""Shift a mesh.

        Args:
            args (argparse.Namespace): Parsed arguments.
            mesh (ObjDict): Mesh to shift.
            x (float): Amount to shift the plant in the x direction.
            y (float): Amount to shift the plant in the y direction.
            plantid (int, optional): Amount that colors should be shifted
                in the blue channel to account for plant ID.

        Returns:
            ObjDict: Shifted mesh.

        """
        xo = x.to(args.mesh_units).value
        yo = y.to(args.mesh_units).value
        return utils.shift_mesh(mesh, xo, yo, plantid=plantid,
                                axis_up=args.axis_up,
                                axis_x=args.axis_rows)

    @classmethod
    def merge_mesh(cls, args, output):
        r"""Merge the output for multiple IDs.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (list): Set of meshes to merge according to args.

        Returns:
            object: Generated output.

        """
        mesh = output[0]
        x = 0.0 * args.x
        y = 0.0 * args.y
        plantid = 0
        for imesh in output[1:]:
            x += args.row_spacing * (args.nrows + 2)
            plantid += (args.nrows * args.ncols)
            mesh.append(
                cls.shift_mesh(args, imesh, x, y, plantid=plantid)
            )
        return mesh

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
        if name not in ['generate', 'geometryids']:
            return super(GenerateTask, self)._merge_output(
                name, output, merged_param)
        assert self.args.id == 'all_combined'
        if name == 'generate':
            return self.merge_mesh(self.args, list(output.values()))
        elif name == 'geometryids':
            values = list(output.values())
            components = values[0][0]
            geometryids = []
            plantid = self.args.plantid
            assert plantid == 0
            for x in values:
                assert x[0] == components
                x[1]['plantids'] += plantid
                geometryids.append(x[1])
                plantid += (self.args.nrows * self.args.ncols)
            geometryids = {k: np.hstack([x[k] for x in geometryids])
                           for k in geometryids[0].keys()}
            return (components, geometryids)

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id not in ['all', 'all_combined']
        if name in ['generate', 'geometryids']:
            if self.args.canopy == 'single':
                mesh, geometryids = self._generate_single_plant()
            else:
                mesh, geometryids = self._generate_field()
            # Store the unnamed output
            if name == 'generate':
                self.set_output('geometryids', geometryids)
                return mesh
            else:
                self.set_output('generate', mesh)
                return geometryids
        return super(GenerateTask, self)._generate_output(name)

    def generate_single_plant(self, **kwargs):
        r"""Generate a 3D mesh for a single plant.

        Args:
            **kwargs: Any keyword arguments are used to update the task
                arguments.

        Returns:
            ObjDict: Generated mesh.

        """
        kwargs.setdefault('canopy', 'single')
        return self.run_iteration(args_overwrite=kwargs,
                                  output_name='generate')

    def _generate_single_plant(self):
        r"""Generate a 3D mesh for a single plant.

        Returns:
            ObjDict: Generated mesh.

        """
        from openalea.lpy import Lsystem
        x = self.args.x
        y = self.args.y
        plantid = self.args.plantid
        self.log(f'generate_single_plant: {x}, {y}, {plantid}')
        if ((((x.value > 0) or (y.value > 0))
             and (plantid == 0 or self.args.save_all_plantids))):
            inst = self.run_iteration(
                args_overwrite={'x': 0.0, 'y': 0.0},
            )
            mesh = self.shift_mesh(
                self.args, inst.get_output('generate'), x, y, plantid,
            )
            return (mesh, inst.get_output('geometryids'))
        assert 'RUNTIME_PARAM' not in self.parameters
        assert self.args.age is not None
        if isinstance(self.args.age.crop_age_string, str):
            age = ParametrizeCropTask.get_age_from_parameters(
                self.args.age.crop_age_string, self.parameters)
        else:
            age = self.args.age.value
        derivation_length = int(min(
            self.parameters['NMax'] * self.parameters['Plastocron']
            / self.parameters['AgeInc'],
            age / self.parameters['AgeInc']
        ))
        lpy_param = {
            'RUNTIME_PARAM': dict(
                self.parameters,
                seed=plantid,
                verbose_lpy=self.args.verbose_lpy,
                no_class_defaults=True,
                **self.runtime_parameters
            ),
            'OUTPUT_TIME': self.parameter_unit_system.convert(
                age, strip=True,
            ),
            'DERIVATION_LENGTH': derivation_length,
        }
        if self.args.color.string == 'plantids':
            color = [0, 255, plantid]
        elif self.args.color.string == 'componentids':
            raise NotImplementedError
        else:
            color = self.args.color.value
        lpy_model = self.output_file('lpy_model')
        if not os.path.isfile(lpy_model):
            self.get_output('lpy_model')
            assert os.path.isfile(lpy_model)
        lsys = Lsystem(lpy_model, lpy_param)
        tree = lsys.derive()
        # print(f"LSTRING: {tree}")
        generator = lsys.context().globals()['generator']
        scene = lsys.sceneInterpretation(tree)
        lsys_units = generator.unit_system
        components = {}
        mesh = utils.scene2geom(
            scene, self.args.mesh_format,
            axis_up=self.args.axis_up,
            axis_x=self.args.axis_rows,
            color=color,
            components=components,
        )
        mesh = utils.scale_mesh(mesh, 1.0,
                                from_units=lsys_units.length,
                                to_units=self.args.mesh_units)
        if x.value > 0 or y.value > 0:
            mesh = self.shift_mesh(
                self.args, mesh, x, y,
            )
        plantids = plantid * np.ones(
            (mesh.count_elements('face'), ), dtype=np.uint32)
        componentids = np.zeros(
            (mesh.count_elements('face'), ), dtype=np.uint32)
        component_order = sorted(list(components.keys()))
        for i, k in enumerate(component_order):
            for prev, count in components[k]:
                componentids[prev:(prev + count)] = i
        geometryids = (
            component_order, {
                'plantids': plantids, 'componentids': componentids,
            }
        )
        return (mesh, geometryids)

    def generate_field(self, **kwargs):
        r"""Generate a 3D mesh for a field of plants.

        Args:
            **kwargs: Any keyword arguments are used to update the task
                arguments.

        Returns:
            ObjDict: Generated mesh.

        """
        kwargs.setdefault('canopy', 'unique')
        return self.run_iteration(args_overwrite=kwargs,
                                  output_name='generate')

    def _generate_field(self):
        r"""Generate a 3D mesh for a field of plants.

        Returns:
            ObjDict: Generated mesh.

        """
        self.get_parameters()
        x = self.args.x
        y = self.args.y
        plantid = self.args.plantid
        geometryids = []
        components = None
        self.log(f'generate_field: {x}, {y}, {plantid}')
        # Generate the unshifted field so it can be reused
        if x.value > 0 or y.value > 0 or plantid > 0:
            inst = self.run_iteration(
                args_overwrite={'x': 0.0, 'y': 0.0, 'plantid': 0},
            )
            mesh = self.shift_mesh(
                self.args, inst.get_output('generate'), x, y, plantid,
            )
            icomponents, igeometryids = inst.get_output('geometryids')
            if components is None:
                components = icomponents
            igeometryids['plantids'] += plantid
            geometryids.append(igeometryids)
            geometryids = {k: np.hstack([x[k] for x in geometryids])
                           for k in geometryids[0].keys()}
            geometryids = (components, geometryids)
            return (mesh, geometryids)

        generator = np.random.default_rng(seed=plantid)

        def posdev():
            return parse_quantity(generator.normal(
                0.0,
                self.args.location_stddev * self.args.plant_spacing.value
                ), self.args.plant_spacing.units
            )

        # First plant
        inst = self.run_iteration(
            args_overwrite={
                'x': x, 'y': y, 'plantid': plantid,
                'canopy': 'single', 'nrows': 1, 'ncols': 1,
            },
        )
        mesh = inst.get_output('generate')
        icomponents, igeometryids = inst.get_output('geometryids')
        if components is None:
            components = icomponents
        geometryids.append(igeometryids)
        plantid += 1

        # Remainder of field
        if self.args.canopy == 'unique':
            inst_prev = inst
            for i in range(self.args.nrows):
                ix = i * self.args.row_spacing
                for j in range(self.args.ncols):
                    iy = j * self.args.plant_spacing
                    if i == 0 and j == 0:
                        continue
                    inst = self.run_iteration(
                        args_overwrite={
                            'x': ix + posdev(), 'y': iy + posdev(),
                            'plantid': plantid, 'canopy': 'single',
                            'nrows': 1, 'ncols': 1,
                        },
                        copy_outputs_from=inst_prev,
                    )
                    mesh.append(inst.get_output('generate'))
                    geometryids.append(inst.get_output('geometryids')[-1])
                    plantid += 1
                    inst_prev = inst
        elif self.args.canopy == 'tile':
            mesh_single = type(mesh)(mesh)
            for i in range(self.args.nrows):
                ix = i * self.args.row_spacing
                for j in range(self.args.ncols):
                    iy = j * self.args.plant_spacing
                    if i == 0 and j == 0:
                        continue
                    mesh.append(
                        self.shift_mesh(
                            self.args, mesh_single,
                            ix + posdev(), iy + posdev(),
                            plantid=plantid,
                        )
                    )
                    igeometryids = copy.deepcopy(geometryids[0])
                    igeometryids['plantids'][:] = plantid
                    geometryids.append(igeometryids)
                    plantid += 1
        else:
            raise ValueError(
                f"Unsupported canopy type: {self.args.canopy}")
        geometryids = {k: np.hstack([x[k] for x in geometryids])
                       for k in geometryids[0].keys()}
        geometryids = (components, geometryids)
        return (mesh, geometryids)


__all__ = ["monocot", "maize", "dicot", "tomato"]
