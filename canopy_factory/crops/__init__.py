import os
import copy
import numpy as np
from datetime import datetime
from openalea.lpy import Lsystem
from yggdrasil import units, rapidjson
from canopy_factory import utils
from canopy_factory.utils import (
    parse_units, parse_quantity, parse_solar_time, parse_axis, parse_color,
    get_class_registry, UnitSet,
    cached_property, readonly_cached_property,
)
from canopy_factory.cli import TaskBase
from canopy_factory.crops import monocot, maize


############################################################
# TASKS
############################################################

class ParametrizeCropTask(TaskBase):
    r"""Class for generating the LSystem parameters for a canopy."""

    _name = 'parametrize'
    _ext = '.json'
    _help = 'Generate the parameters for an LSystem crop model.'
    _output_dir = utils._param_dir
    _arguments_suffix_ignore = [
        'crop', 'id', 'update_lpy_model', 'data', 'data_length_units',
    ]
    _arguments = [
        (('--update-lpy-model', ), {
            'type': str, 'const': True, 'nargs': '?',
            'help': ('File containing LPy L-system rules that should be '
                     'updated with default parameters'),
        }),
        (('--data', ), {
            'type': str,
            'help': ('File containing raw data that should be used '
                     'to set parameters'),
        }),
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

    @staticmethod
    def add_arguments_static(cls, parser, only_subparser=False,
                             only_crop_parameters=False, **kwargs):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            only_subparser (bool, optional): If True, only add the
                subparser if it is missing.
            only_crop_parameters (list, optional): Set of crop parameters
                that should be added to the parser.
            **kwargs: Additional keyword arguments are passed to the
                parent class method.

        """
        TaskBase.add_arguments_static(
            cls, parser, only_subparser=True, **kwargs
        )
        subparser = parser.get_subparser(cls._registry_key, cls._name)
        for v in get_class_registry().values('crop'):
            v.add_arguments(subparser, only_subparser=True, **kwargs)
        if only_subparser:
            return
        TaskBase.add_arguments_static(
            cls, parser, only_subparser=only_subparser, **kwargs
        )
        for k, v in get_class_registry().items('crop'):
            v.add_arguments(subparser, only_subparser=only_subparser,
                            only_crop_parameters=only_crop_parameters,
                            **kwargs)

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        super(ParametrizeCropTask, cls).adjust_args(args)
        if args.update_lpy_model is True:
            args.update_lpy_model = os.path.join(
                utils._lpy_dir, f'{args.crop}.lpy')
        if (not args.data) and args.crop:
            generator = get_class_registry().get('crop', args.crop)
            args.data = generator._default_data

    @classmethod
    def output_base(cls, args, name=None):
        r"""Generate the base file name that should be used to generate
        an output file name.

        Args:

            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: File base.

        """
        return f'{args.crop}_{args.id}'

    @classmethod
    def output_suffix(cls, args, name=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        return ''

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        with open(outputfile, 'w') as fd:
            rapidjson.dump(output, fd, write_mode=rapidjson.WM_PRETTY)

    @cached_property
    def generator(self):
        r"""PlantGenerator: Plant generator that should be parameterized."""
        return get_class_registry().get('crop', self.args.crop)

    @cached_property
    def parameters(self):
        r"""dict: Set of model parameters collected from the command line."""
        kwargs = {}
        for k, v in self.generator._arguments:
            kattr = self.arg2dest(k, v)
            if getattr(self.args, kattr, None) is not None:
                kwargs[kattr] = getattr(self.args, kattr)
        for k in ['verbose', 'debug', 'debug_param',
                  'debug_param_prefix']:
            kwargs[k] = getattr(self.args, k)
        inst = self.generator(**kwargs)
        out = copy.deepcopy(inst.all_parameters)
        if self.args.data:
            self.generator.parameters_from_file(self.args, out)
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
        with open(self.args.update_lpy_model, 'r') as fd:
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
            'generator = generator_cls(**RUNTIME_PARAM)'
        ]
        contents = [
            prefix_contents, comment_start
        ] + contents + [
            comment_end, suffix_contents
        ]
        with open(self.args.update_lpy_model, 'w') as fd:
            fd.write('\n'.join(contents))

    @classmethod
    def _run(cls, self):
        if self.args.update_lpy_model:
            self.add_parameters_to_lpy_model()
        generator_args = copy.deepcopy(self.parameters)
        out = {
            'RUNTIME_PARAM': generator_args,
        }
        return out


class LayoutTask(TaskBase):
    r"""Class for plotting the layout of a canopy."""

    _name = 'layout'
    _ext = '.png'
    _output_dir = os.path.join(utils._output_dir, 'layout')
    _time_vars = ['time']
    _hour_defaults = {}  # 'time': 12}
    _arguments_suffix_ignore = [
        'locaton', 'time', 'doy', 'hour', 'year', 'timezone',
    ]
    _convert_to_mesh_units = [
        'plot_length', 'plot_width', 'row_spacing', 'plant_spacing',
        'x', 'y',
        'ground_height',
    ]
    _convert_to_color_tuple = []
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
            'type': int, 'default': 2,
            'help': ('Number of times the canopy should be repeated in '
                     'each direction'),
        }),
        (('--location', ), {
            'type': str, 'default': 'Champaign',
            'choices': sorted(list(utils.read_locations().keys())),
            'help': ('Name of a registered location that should be used '
                     'to set the location dependent properties: '
                     'timezone, altitude, longitude, latitude'),
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
        (('--latitude', '--lat', ), {
            'type': parse_quantity,
            'default': 40.1164, 'units': 'degrees',
            'help': ('Latitude (in degrees) at which the sun should be '
                     'modeled. Defaults to the latitude of Champaign '
                     'IL.'),
        }),
        (('--altitude', '--elevation', ), {
            'type': parse_quantity,
            'default': 224.0, 'units': 'meters',
            'help': ('Altitude (in meters) that should be used for '
                     'solar light calculations. If not provided, it '
                     'will be calculated from \"pressure\", if it is '
                     'provided, and the elevation of Champaign, IL '
                     'otherwise.'),
        }),
        (('--pressure', ), {
            'type': parse_quantity, 'units': 'Pa',
            'help': ('Air pressure (in Pa) that should be used for '
                     'solar light calculations. If not provided, it '
                     'will be calculated from \"altitude\".'),
        }),
        (('--temperature', ), {
            'type': parse_quantity, 'default': 12.0, 'units': 'degC',
            'help': ('Air temperature (in degrees C) that should be '
                     'used for solar light calculations.'),
        }),
        (('--longitude', '--long', ), {
            'type': parse_quantity,
            'default': -88.2434, 'units': 'degrees',
            'help': ('Longitude (in degrees) at which the sun should be '
                     'modeled. Defaults to the longitude of Champaign '
                     'IL.'),
        }),
        (('--time', '-t', ), {
            'type': str,  # 'default': '2024-06-17',
            'help': ('Date time (in any ISO 8601 format) that the sun '
                     'should be modeled for. If hour information is not '
                     'provided, the provided \"hour\" will be used. '
                     'If \"now\" is specified the current date and time '
                     'will be used.'),
        }),
        (('--doy', ), {
            'type': int,
            'help': ('Day of the year that the sun should be modeled '
                     'for.'),
        }),
        (('--hour', '--hr', ), {
            'type': int,
            'help': ('Hour that the sun should be modeled for. If '
                     'provided with \"--time\", any hour information in '
                     'the specified time will be overwritten. Defaults '
                     'to 12 if \"--doy\" is provided, but \"--hour\" is '
                     'not.'),
        }),
        (('--year', ), {
            'type': int,
            'help': ('Year that sun should be modeled for. If provided '
                     'with \"--time\" (or \"--start-time\"/'
                     '\"--stop-time\"), the year in the time string(s) '
                     'will be overwritten. Defaults to the current year '
                     'if \"--doy\" is provided, but \"--year\" is not.'),
        }),
        (('--timezone', '--tz', ), {
            'type': str,
            'help': ('Name of timezone (as accepted by pytz) for '
                     'location that sun should be modeled. If provided '
                     'with \"--time\" (or \"--start-time\"/'
                     '\"--stop-time\"), any timezone information in the '
                     'specified time(s) will be overwritten. Defaults '
                     'to \"America/Chicago\" if \"--doy\" is provided, '
                     'but \"--timezone\" is not.'),
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
    _argument_modifications = {
        '--output': {
            'help': 'File where the layout should be saved',
        },
    }

    @staticmethod
    def _on_registration(cls):
        TaskBase._on_registration(cls)
        if cls._registry_key is None or cls._name is None:
            return
        import inspect
        base = inspect.getmro(cls)[1]
        cls._convert_to_mesh_units = cls.select_valid_arguments(
            getattr(base, '_convert_to_mesh_units', [])
            + cls._convert_to_mesh_units)
        cls._convert_to_color_tuple = cls.select_valid_arguments(
            getattr(base, '_convert_to_color_tuple', [])
            + cls._convert_to_color_tuple)

    @classmethod
    def _read_output(cls, args, name=None):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            object: Contents of the output file.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        return utils.read_png(outputfile, verbose=args.verbose)

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        output.savefig(outputfile, dpi=300)

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        for k, v in cls._excluded_arguments_defaults.items():
            if not hasattr(args, k):
                setattr(args, k, v)
        if isinstance(args.mesh_units, str):
            args.mesh_units = units.Units(args.mesh_units)
        for k in cls._convert_to_mesh_units:
            setattr(args, k, parse_quantity(getattr(args, k, None),
                                            args.mesh_units))
        if args.canopy == 'single':
            args.nrows = 1
            args.ncols = 1
        else:
            if args.plot_width is None:
                args.plot_width = args.nrows * args.row_spacing
            args.nrows = int(args.plot_width / args.row_spacing)
            args.ncols = int(args.plot_length / args.plant_spacing)
        # args.axis_cols = np.cross(args.axis_rows, args.axis_up)
        args.axis_cols = np.cross(args.axis_up, args.axis_rows)
        args.axis_east = np.cross(args.axis_north, args.axis_up)
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
        if args.location:
            location_data = utils.read_locations()
            for k, v in location_data[args.location].items():
                setattr(args, k, v)
        if not (args.pressure or args.altitude):
            args.altitude = parse_quantity(10.0, 'meters')
        cls.adjust_args_time(args)
        super(LayoutTask, cls).adjust_args(args)
        for k in cls._convert_to_color_tuple:
            v = getattr(args, k, None)
            if isinstance(v, str):
                setattr(args, f'{k}_str', v)
                setattr(args, k, parse_color(v, convert_names=True))

    @classmethod
    def adjust_args_time(cls, args, timevar=None):
        r"""Adjust the time related variables in a set of parsed
        arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            timevar (str, optional): Time variable to adjust. If not
                provided, all of the time variables associated with this
                subparser will be adjusted.

        """
        if timevar is None:
            for tv in cls._time_vars:
                cls.adjust_args_time(args, timevar=tv)
            return
        x = getattr(args, timevar)
        x_str = None
        x_solar = None
        if x in utils._solar_times:
            x_solar = x
            if args.doy:
                x = None
            else:
                x = '2024-06-17'
        if x:
            if isinstance(x, datetime):
                pass
            else:
                x = datetime.fromisoformat(x)
            if not (x.tzinfo or args.timezone):
                args.timezone = "America/Chicago"
            if not (x.hour or args.hour):
                args.hour = cls._hour_defaults.get(timevar, None)
            if not (x.year or args.year):
                args.year = datetime.now().year
        elif args.doy:
            if not args.hour:
                args.hour = cls._hour_defaults.get(timevar, None)
            if not args.year:
                args.year = datetime.now().year
            if not args.timezone:
                args.timezone = "America/Chicago"
            x = datetime.strptime(args.year, args.doy, "%Y-%j")
        if isinstance(args.timezone, str):
            import pytz
            args.timezone = pytz.timezone(args.timezone)
        if x:
            replacements = {}
            if args.hour:
                replacements['hour'] = args.hour
                args.hour = None
            if args.year:
                replacements['year'] = args.year
                args.year = None
            if replacements:
                x = x.replace(**replacements)
            if args.timezone:
                x = x.astimezone(args.timezone)
                args.timezone = None
        if x_solar in utils._solar_times:
            date = x.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            x = parse_solar_time(
                x_solar, date, args.latitude, args.longitude,
                altitude=args.altitude,
            )
            x_str = date.date().isoformat() + '-' + x_solar
            assert ':' not in x_str
        if x and x != getattr(args, timevar):
            if x_str is None:
                x_str = x.replace(microsecond=0).isoformat().replace(
                    ':', '-')
            setattr(args, timevar, x)
            setattr(args, f'{timevar}_str', x_str)
            # print(f'Updated {timevar} to {x} ({x_str})')

    @classmethod
    def output_suffix(cls, args, name=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            str: Suffix.

        """
        suffix = ''
        if args.canopy != 'single':
            suffix += f'_canopy{args.canopy.title()}'
        if args.location:
            suffix += f"_{args.location}"
        suffix += cls.output_suffix_time(args)
        if args.periodic_canopy:
            suffix += (f'_periodic{args.periodic_canopy_count}'
                       f'_{args.periodic_canopy}')
        return suffix

    @classmethod
    def output_suffix_time(cls, args, timevar=None):
        r"""Get the suffix containing time information that should be
        included in generated output file names.

        Args:
            args (argparse.Namespace): Parsed arguments.
            timevar (str, optional): Time variable to generate a suffix
                for. If not provided, a suffix combining all of the
                time variables associated with this subparser will be
                returned.

        Returns:
            str: Suffix.

        """
        if timevar is None:
            suffixes = [cls.output_suffix_time(args, tv)
                        for tv in cls._time_vars]
            suffixes = [x for x in suffixes if x is not None]
            return '_'.join(suffixes)
        time_str = getattr(args, f'{timevar}_str', None)
        if time_str:
            return time_str
        if getattr(args, timevar) is None:
            return None
        time = getattr(args, timevar).replace(microsecond=0)
        return time.isoformat().replace(':', '-')

    @cached_property
    def nplants(self):
        r"""int: Total number of plants in the canopy."""
        # TODO: Allow all?
        if self.args.canopy == 'single':
            return 1
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
        if self.args.canopy == 'single':
            return (2 * self.args.periodic_canopy_count + 1)**2 - 1
        return self.nplants * (
            (2 * self.args.periodic_canopy_count + 1)**2 - 1)

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
                self.args.periodic_canopy_count * np.ones((3, ), 'i4')),
            pos_units0,
        )
        out = []
        for pos in pos0:
            out.append(shifts + pos)
        out = np.vstack(out)
        assert out.shape[0] == self.nplants_periodic
        assert out.shape[1] == 3
        return units.QuantityArray(out, pos_units0)

    @classmethod
    def parse_time(cls, x, args, timevar=None):
        r"""Adjust the time related variables in a set of parsed
        arguments.

        Args:
            x (str, datetime.datetime): Time to parse.
            args (argparse.Namespace): Parsed arguments.
            timevar (str, optional): Name of the variable that should be
                updated on args.

        Returns:
            datetime.datetime: Parsed time.

        """
        x_str = None
        x_solar = None
        if x in utils._solar_times:
            x_solar = x
            if args.doy:
                x = None
            else:
                x = '2024-06-17'
        if x:
            if isinstance(x, datetime):
                pass
            else:
                x = datetime.fromisoformat(x)
            if not (x.tzinfo or args.timezone):
                args.timezone = "America/Chicago"
            if not (x.hour or args.hour):
                args.hour = cls._hour_defaults.get(timevar, None)
            if not (x.year or args.year):
                args.year = datetime.now().year
        elif args.doy:
            if not args.hour:
                args.hour = cls._hour_defaults.get(timevar, None)
            if not args.year:
                args.year = datetime.now().year
            if not args.timezone:
                args.timezone = "America/Chicago"
            x = datetime.strptime(args.year, args.doy, "%Y-%j")
        if isinstance(args.timezone, str):
            import pytz
            args.timezone = pytz.timezone(args.timezone)
        if x:
            replacements = {}
            if args.hour:
                replacements['hour'] = args.hour
            if args.year:
                replacements['year'] = args.year
            if replacements:
                x = x.replace(**replacements)
            reset = list(replacements.keys())
            if args.timezone:
                x = x.astimezone(args.timezone)
                reset.append('timezone')
            if reset and timevar is not None:
                for k in reset:
                    setattr(args, k, None)
        if x_solar in utils._solar_times:
            date = x.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            x = parse_solar_time(
                x_solar, date, args.latitude, args.longitude,
                altitude=args.altitude,
            )
            x_str = date.date().isoformat() + '-' + x_solar
            assert ':' not in x_str
        if x and timevar is not None and x != getattr(args, timevar):
            if x_str is None:
                x_str = x.replace(microsecond=0).isoformat().replace(
                    ':', '-')
            setattr(args, timevar, x)
            setattr(args, f'{timevar}_str', x_str)
            # print(f'Updated {timevar} to {x} ({x_str})')
        return x

    def get_solar_model(self, time=None):
        if time is None:
            return self.solar_model
        from canopy_factory.raytrace import SolarModel
        if isinstance(time, str):
            time = self.parse_time(time, self.args)
        return SolarModel(
            self.args.latitude, self.args.longitude, time,
            altitude=self.args.altitude, pressure=self.args.pressure,
            temperature=self.args.temperature,
        )

    def get_solar_direction(self, time=None):
        if time is None:
            return self.solar_direction
        return self.get_solar_model(time).relative_direction(
            self.args.axis_up, self.args.axis_north)

    # def get_solar_alititude(self, time=None):
    #     return self.get_solar_model(time).apparent_elevation

    def isExteriorPlant(self, plantid, nbuffer_col=1, nbuffer_row=1):
        r"""Determine if a plant is on the edge of the field.

        Args:
            plantid (int): Plant identifier.
            nbuffer_col (int, optional): Number of plants from the edge
                along the columns to count as exterior.
            nbuffer_row (int, optional): Number of plants from the edge
                along the rowss to count as exterior.

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
        if self.args.time is None:
            return None
        return self.get_solar_model(self.args.time)

    @cached_property
    def solar_direction(self):
        if self.args.time is None:
            return None
        return self.solar_model.relative_direction(
            self.args.axis_up, self.args.axis_north)

    @cached_property
    def solar_elevation(self):
        if self.args.time is None:
            return None
        return self.solar_model.apparent_elevation

    @cached_property
    def scene_layout(self):
        out = {
            'plants': self.project_onto_ground(self.plant_positions),
            'periodic_plants': self.project_onto_ground(
                self.plant_positions_periodic),
            'north': self.project_onto_ground(
                self.args.axis_north, ray=True),
            'east': self.project_onto_ground(
                self.args.axis_east, ray=True),
        }
        if self.solar_direction is not None:
            out.update(
                sun_ray=self.project_onto_ground(
                    -self.solar_direction, ray=True),
                sun_elevation=self.solar_elevation,
            )
        return out

    @cached_property
    def subplot_ratio(self):
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
        if self.args.time:
            self.plot_sun(self.args.time, plant_min, plant_max,
                          arrow_length=arrow_length)
        else:
            for t in ['sunrise', 'sunset', 'noon']:
                self.plot_sun(t, plant_min, plant_max,
                              arrow_length=arrow_length)

    @classmethod
    def _run(cls, self, **kwargs):
        self.plot_layout(self.scene_layout)
        return self.figure


class GenerateTask(LayoutTask):
    r"""Class for generating 3D canopies."""

    _name = 'generate'
    _ext = None
    _help = 'Generate a canopy mesh'
    _output_dir = os.path.join(utils._output_dir, 'meshes')
    _time_vars = []
    _arguments_suffix_ignore = [
        'crop', 'id', 'canopy', 'color',
        'overwrite_lpy_param', 'plantid', 'debug_param',
        'debug_param_prefix',
        'unful_leaves', 'mesh_format', 'overwrite_generate',
        'plot_width', 'output_plantids', 'overwrite_plantids',
    ]
    _alternate_outputs_write_required = ['plantids']
    _convert_to_color_tuple = [
        'color',
    ]
    _arguments = [
        (('--age', ), {
            'type': parse_quantity, 'default': 27, 'units': 'days',
            'help': ('Plant age to generate model for (in days '
                     'since planting)'),
        }),
        (('--data', ), {
            'type': str,
            'help': ('File containing raw data that should be used '
                     'to set parameters'),
        }),
        (('--lpy-input', ), {
            'type': str,
            'help': 'File containing LPy L-system rules',
        }),
        (('--lpy-param', ), {
            'type': str,
            'help': 'File containing parameters for L-system rules',
        }),
        (('--overwrite-lpy-param', ), {
            'action': 'store_true',
            'help': 'Overwrite the existing lpy_param file',
        }),
        (('--niter', ), {
            'type': int, 'default': 20,
            'help': 'Number of iterations to generate',
        }),
        (('--color', ), {
            'type': parse_color, 'default': 'green',
            'help': ('Color that should be used for the generated plant. '
                     'This can be a color name or 3 comma separated RGB '
                     'values expressed as integers in the range '
                     '[0, 255]. If a values of \'plantid\' is provided, '
                     'colors will be used to identify individual plants '
                     'by setting the blue channel to the plant ID.'),
        }),
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
        # (('--unfurl-leaves', ), {
        #     'action': 'store_true',
        #     'help': 'Start leaves as cylinders and then unfurl them',
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
        (('--output-plantids', ), {
            'nargs': '?', 'const': True, 'default': True,
            'help': ('File where the IDs of the plant each face belongs '
                     'to should be saved'),
        }),
        (('--overwrite-plantids', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing plant IDs file '
                     '"--output-plantids" is passed'),
        }),
    ]
    _argument_modifications = {
        '--output': {
            'help': 'File where the generated mesh should be saved',
        },
        '--canopy': {
            'default': 'single',
        },
    }
    _excluded_arguments = [
        '--axis-north', '--ground-height', '--latitude',
        '--altitude', '--pressure', '--temperature', '--longitude',
        '--time', '--doy', '--hour', '--year', '--timezone',
        '--interior-plant-color', '--exterior-plant-color',
        '--periodic-plant-color',
        '--periodic-canopy', '--periodic-canopy-count',
    ]

    @staticmethod
    def add_arguments_static(cls, parser, **kwargs):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            **kwargs: Additional keyword arguments are passed to the
                parent class method.

        """
        ParametrizeCropTask.add_arguments_static(
            cls, parser, only_crop_parameters=['id', 'LeafUnfurled'],
            **kwargs)
        subparser = parser.get_subparser(cls._registry_key, cls._name)
        for k in get_class_registry().keys('crop'):
            parser = subparser.get_subparser('crop', k)
            action = parser.find_argument('id')
            if not action.choices:
                action.choices = []
            action.choices += ['all', 'all_split']

    @classmethod
    def _read_output(cls, args, name=None):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            object: Contents of the output file.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        if name == 'plantids':
            return utils.read_csv(outputfile, select='plantids')
        return utils.read_3D(outputfile, file_format=args.mesh_format,
                             verbose=args.verbose)

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if args.id == 'all_split':
            return
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        if name == 'plantids':
            return utils.write_csv({'plantids': output}, outputfile)
        utils.write_3D(output, outputfile, file_format=args.mesh_format,
                       verbose=args.verbose)

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.save_all_plantids = False
        if not args.output_generate:
            if not args.mesh_format:
                args.mesh_format = 'obj'
        super(GenerateTask, cls).adjust_args(args)
        args.plantids_in_blue = False
        if not args.mesh_format:
            args.mesh_format = utils.get_3D_format(args.output_generate)
        if (not args.data) and args.crop:
            generator = get_class_registry().get('crop', args.crop)
            args.data = generator._default_data
        if args.color_str == 'plantids':
            args.plantids_in_blue = True
        if args.lpy_input is None:
            args.lpy_input = os.path.join(
                utils._lpy_dir, f'{args.crop}.lpy')
        if args.lpy_param is None:
            args.lpy_param = os.path.join(
                utils._param_dir, f'{args.crop}_{args.id}.json')

    @cached_property
    def lpy_param(self):
        r"""dict: Runtime parameters for the lpy model."""
        if not (self.args.lpy_param
                and os.path.isfile(self.args.lpy_param)):
            return {}
        with open(self.args.lpy_param, 'r') as fd:
            out = rapidjson.load(fd)
        return out

    @classmethod
    def output_base(cls, args, name=None):
        r"""Generate the base file name that should be used to generate
        an output file name.

        Args:

            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: File base.

        """
        return f'{args.crop}_{args.id}'

    @classmethod
    def output_suffix(cls, args, name=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        suffix = ''
        if args.canopy != 'single':
            suffix += f'_canopy{args.canopy.title()}'
        # if args.LeafUnfurled:
        #     suffix += '_unfurled'
        if name not in ['plantids', 'layout']:
            color_str = None
            if isinstance(args.color, str):
                color_str = args.color
            elif getattr(args, 'color_str', None):
                color_str = args.color_str
            elif args.color:
                return False
            if color_str != 'green':
                suffix += f'_{color_str}'
        if args.plantid > 0:
            if args.save_all_plantids:
                suffix += f'_{args.plantid}'
            else:
                return False
        return suffix

    @classmethod
    def output_ext(cls, args, name=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Output file extension.

        """
        if name == 'plantids':
            return '.csv'
        ext = super(GenerateTask, cls).output_ext(args, name=name)
        if ext is None and args.mesh_format:
            ext = utils._inv_geom_ext[args.mesh_format]
        return ext

    @cached_property
    def generator(self):
        r"""PlantGenerator: Plant generator that should be parameterized."""
        return get_class_registry().get('crop', self.args.crop)

    @readonly_cached_property
    def all_ids(self):
        r"""list: All crop classes for current data."""
        return self.generator.ids_from_file(self.args.data)

    @classmethod
    def _run(cls, self):
        r"""Run the process associated with this subparser."""
        if self.args.id == 'all_split':
            plantid = self.args.plantid
            x = self.args.x
            y = self.args.y
            for i, id in enumerate(self.all_ids):
                cls.run_class(
                    self, dont_reset_alternate_output=True,
                    dont_load_existing=True,
                    args_overwrite={
                        'x': x, 'y': y, 'plantid': plantid,
                        'id': id,
                        'lpy_param': None,
                    },
                )
            mesh = None
            self.add_alternate_output('plantids', None)
        elif self.args.id == 'all':
            mesh = None
            plantids = self.pop_alternate_output('plantids', None)
            if plantids is None:
                plantids = []
            else:
                plantids = [plantids]
            plantid = self.args.plantid
            x = self.args.x
            y = self.args.y
            for i, id in enumerate(self.all_ids):
                print("IMESH", i, id)
                imesh = cls.run_class(
                    self, dont_reset_alternate_output=True,
                    args_overwrite={
                        'x': x, 'y': y, 'plantid': plantid,
                        'id': id,
                        'lpy_param': None,
                    },
                )
                if i == 0:
                    mesh = imesh
                else:
                    mesh.append(imesh)
                x += self.args.row_spacing * (self.args.nrows + 2)
                plantid += (self.args.nrows * self.args.ncols)
                plantids.append(self.pop_alternate_output('plantids'))
                # TODO: Labels
            self.add_alternate_output('plantids', np.hstack(plantids))
        elif self.args.canopy == 'single':
            mesh = cls._generate_single_plant(self)
        else:
            mesh = cls._generate_field(self)
        return mesh

    def shift_mesh(self, mesh, x, y, plantid=None):
        r"""Shift a mesh.

        Args:
            mesh (ObjDict): Mesh to shift.
            x (float): Amount to shift the plant in the x direction.
            y (float): Amount to shift the plant in the y direction.
            plantid (int, optional): Amount that colors should be shifted
                in the blue channel to account for plant ID.

        Returns:
            ObjDict: Shifted mesh.

        """
        xo = x.to(self.args.mesh_units).value
        yo = y.to(self.args.mesh_units).value
        return utils.shift_mesh(mesh, xo, yo, plantid=plantid,
                                axis_up=self.args.axis_up,
                                axis_x=self.args.axis_rows,
                                plantids_in_blue=self.args.plantids_in_blue)

    def generate_single_plant(self, **kwargs):
        r"""Generate a 3D mesh for a single plant.

        Args:
            **kwargs: Any keyword arguments are used to update the task
                arguments.

        Returns:
            ObjDict: Generated mesh.

        """
        kwargs.setdefault('canopy', 'single')
        return self.run_class(self, args_overwrite=kwargs)

    @classmethod
    def _generate_single_plant(cls, self):
        r"""Generate a 3D mesh for a single plant.

        Args:
            self (object): Task instance that is running.

        Returns:
            ObjDict: Generated mesh.

        """
        x = self.args.x
        y = self.args.y
        plantid = self.args.plantid
        self.log(f'generate_single_plant: {x}, {y}, {plantid}', cls=cls)
        if ((((x.value > 0) or (y.value > 0))
             and (plantid == 0 or self.args.save_all_plantids))):
            mesh = self.shift_mesh(
                cls.run_class(
                    self, args_overwrite={'x': 0.0, 'y': 0.0},
                ), x, y, plantid,
            )
            return mesh
        self.lpy_param['RUNTIME_PARAM'].update(
            seed=plantid,
            verbose=self.args.verbose,
            debug=self.args.debug,
            debug_param=self.args.debug_param,
            debug_param_prefix=self.args.debug_param_prefix,
            no_class_defaults=True,
        )
        color = self.args.color
        if self.args.color_str == 'plantids':
            color = [0, 255, plantid]
        lsys = Lsystem(self.args.lpy_input, self.lpy_param)
        tree = lsys.axiom
        for i in range(self.args.niter):
            tree = lsys.iterate(tree, 1)
        lsys_units = lsys.context().globals()['generator'].unit_system
        scene = lsys.sceneInterpretation(tree)
        mesh = utils.scene2geom(
            scene, self.args.mesh_format,
            axis_up=self.args.axis_up,
            axis_x=self.args.axis_rows,
            color=color,
        )
        mesh = utils.scale_mesh(mesh, 1.0,
                                from_units=lsys_units.length,
                                to_units=self.args.mesh_units)
        if x.value > 0 or y.value > 0:
            mesh = self.shift_mesh(
                mesh, x, y,
            )
        plantids = plantid * np.ones(
            (mesh.count_elements('face'), ), dtype=np.uint32)
        self.add_alternate_output('plantids', plantids)
        return mesh

    def generate_field(self, **kwargs):
        r"""Generate a 3D mesh for a field of plants.

        Args:
            **kwargs: Any keyword arguments are used to update the task
                arguments.

        Returns:
            ObjDict: Generated mesh.

        """
        kwargs.setdefault('canopy', 'unique')
        return self.run_class(self, args_overwrite=kwargs)

    @classmethod
    def _generate_field(cls, self):
        r"""Generate a 3D mesh for a field of plants.

        Args:
            self (object): Task instance that is running.

        Returns:
            ObjDict: Generated mesh.

        """
        x = self.args.x
        y = self.args.y
        plantid = self.args.plantid
        plantids = self.pop_alternate_output('plantids', None)
        if plantids is None:
            plantids = []
        else:
            plantids = [plantids]
        self.log(f'generate_field: {x}, {y}, {plantid}', cls=cls)
        # Generate the unshifted field so it can be reused
        if x.value > 0 or y.value > 0 or plantid > 0:
            mesh = self.shift_mesh(
                cls.run_class(
                    self, dont_reset_alternate_output=True,
                    args_overwrite={'x': 0.0, 'y': 0.0, 'plantid': 0},
                ), x, y, plantid,
            )
            plantids.append(self.pop_alternate_output('plantids')
                            + plantid)
            self.add_alternate_output('plantids', np.hstack(plantids))
            return mesh

        generator = np.random.default_rng(seed=plantid)

        def posdev():
            return parse_quantity(generator.normal(
                0.0,
                self.args.location_stddev * self.args.plant_spacing.value
                ), self.args.plant_spacing.units
            )

        # First plant
        mesh = cls.run_class(
            self, dont_reset_alternate_output=True,
            args_overwrite={
                'x': x, 'y': y, 'plantid': plantid,
                'canopy': 'single', 'nrows': 1, 'ncols': 1,
            },
        )
        plantids.append(self.pop_alternate_output('plantids'))
        plantid += 1

        # Remainder of field
        if self.args.canopy == 'unique':
            for i in range(self.args.nrows):
                ix = i * self.args.row_spacing
                for j in range(self.args.ncols):
                    iy = j * self.args.plant_spacing
                    if i == 0 and j == 0:
                        continue
                    mesh.append(cls.run_class(
                        self, dont_reset_alternate_output=True,
                        args_overwrite={
                            'x': ix + posdev(), 'y': iy + posdev(),
                            'plantid': plantid, 'canopy': 'single',
                            'nrows': 1, 'ncols': 1,
                        },
                    ))
                    plantids.append(
                        self.pop_alternate_output('plantids'))
                    plantid += 1
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
                            mesh_single, ix + posdev(), iy + posdev(),
                            plantid=plantid,
                        )
                    )
                    plantids.append(
                        plantid
                        * np.ones((mesh_single.count_elements('face'), ),
                                  dtype=np.uint32)
                    )
                    plantid += 1
        else:
            raise ValueError(
                f"Unsupported canopy type: {self.args.canopy}")
        self.add_alternate_output('plantids', np.hstack(plantids))
        return mesh


__all__ = ["monocot", "maize"]
