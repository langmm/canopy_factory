import os
import copy
import numpy as np
import yggdrasil_rapidjson as rapidjson
from yggdrasil_rapidjson import units
from canopy_factory import utils, arguments
from canopy_factory.utils import (
    parse_units, parse_quantity, parse_axis, parse_color,
    get_class_registry, UnitSet,
    cached_property,
)
from canopy_factory.cli import (
    TaskBase, TimeArgument, OutputArgument,
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
            'directory': utils.cfg['directories']['param'],
            'ext': '.json',
            'description': (
                'parameters used to generate 3D '
                'representations of a canopy'
            ),
            'composite_param': ['id', 'data_year'],
        },
        'lpy_model': {
            'directory': utils.cfg['directories']['lpy'],
            'ext': '.lpy',
            'description': 'LPy L-system rules',
        },
    }
    _arguments = [
        arguments.ClassSubparserArgumentDescription(
            'crop', dont_create=True,
            include=['id', 'data', 'data_year'],
            modifications={
                'id': {
                    'append_choices': ['default', 'all'],
                    'suffix_param': {
                        'noteq': 'default',
                        'skip_outputs': ['lpy_model'],
                    },
                },
                'data_year': {
                    'append_choices': ['all'],
                    'suffix_param': {
                        'skip_outputs': ['lpy_model'],
                    },
                },
            },
            suffix_param={
                'cond': True,
            },
        ),
        arguments.ArgumentDescriptionSet([
            (('--piecewise-param', ), {
                'type': str, 'nargs': '?', 'const': 'N',
                'help': (
                    'Independent variable that should be used to '
                    'determine when different parameters should be used'
                ),
                'suffix_param': {},
            }),
            (('--piecewise-param-values', ), {
                'type': float, 'nargs': '+',
                'help': (
                    'Independent variable values at which parameters sets '
                    'should be changed'
                ),
                'suffix_param': {'sep': '-'},
            }),
            (('--piecewise-files', ), {
                'type': str, 'nargs': '+', 'action': 'extend',
                'help': (
                    'Names of parameter files that should be used '
                    'conditionally in the order they should be used'
                ),
                'suffix_param': {'sep': '-'},
            }),
            (('--piecewise-ids', ), {
                'type': str, 'nargs': '+', 'action': 'extend',
                'help': (
                    'IDs for parameters that should be used '
                    'conditionally in the order they should be '
                    'used (if parameter files not provided)'
                ),
                'suffix_param': {'sep': '-'},
            }),
        ], name='piecewise_properties', suffix_param={
            'cond': lambda x: bool(x.piecewise_param),
            'outputs': ['parametrize'],
            'skip_outputs': ['lpy_model'],
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

    def __init__(self, *args, **kwargs):
        super(ParametrizeCropTask, self).__init__(*args, **kwargs)
        if ((isinstance(getattr(self.args, 'id', None), str)
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
                getattr(self.args, 'data', None), crop=self.args.crop,
                year=(None if (isinstance(self.args.data_year, str)
                               and self.args.data_year.startswith('all'))
                      else self.args.data_year),
            )
        self.finalize()

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        super(ParametrizeCropTask, cls).adjust_args_internal(args, **kwargs)
        idstr = args.id
        if isinstance(idstr, str) and idstr.startswith('all'):
            idstr = None
        fdata = None
        if args.data:
            fdata = utils.DataProcessor.from_file(args.data)
            args.data_year = fdata.year
        elif args.data_year and not args.data_year.startswith('all'):
            args.data = utils.DataProcessor.output_name(args.crop,
                                                        args.data_year)
            if ((utils.DataProcessor._ignore_data
                 or not os.path.isfile(args.data))):
                args.data = None
                args.data_year = None
            else:
                fdata = utils.DataProcessor.from_file(args.data)
        if ((idstr is not None and fdata is not None
             and idstr not in fdata.ids)):
            args.data = None
            args.data_year = None
        if not args.data_year:
            years = utils.DataProcessor.available_years(
                args.crop, id=idstr)
            if years:
                args.data_year = years[0]
                args.data = utils.DataProcessor.output_name(
                    args.crop, args.data_year)
        if idstr is not None:
            if args.piecewise_param:
                if args.piecewise_files:
                    args.piecewise_ids = None
                elif not args.piecewise_ids:
                    id_base = idstr.split('_')[0]
                    ids = utils.DataProcessor.available_ids(
                        args.crop, year=args.data_year)
                    args.piecewise_ids = [
                        x for x in ids
                        if x.startswith(id_base) and x != idstr
                    ]
            else:
                args.piecewise_ids = []
                args.piecewise_files = []
                args.piecewise_values = []

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
        kwargs = self.generator_class._arguments.extract_args(self.args)
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

    def generate_piecewise_merge(self):
        r"""Generate a table of piecewise parameters."""
        out = copy.deepcopy(
            self.run_iteration(
                args_overwrite={
                    'piecewise_param': None,
                    'output_parametrize': True,
                },
                output_name='parametrize',
            )
        )
        merge = []
        if self.args.piecewise_files:
            for x in self.args.piecewise_files:
                merge.append(self.read_output('parametrize', x))
        else:
            for iid in self.args.piecewise_ids:
                merge.append(self.run_iteration(
                    args_overwrite={
                        'id': iid,
                        'piecewise_param': None,
                        'output_parametrize': True,
                    },
                    output_name='parametrize',
                ))
        merged = {}
        for iout in merge:
            for k in list(iout.keys()):
                kequal = (iout[k] == out.get(k, None))
                if isinstance(kequal, np.ndarray):
                    kequal = kequal.all()
                if kequal:
                    del iout[k]
                elif k == 'id':
                    out[k] += '_' + iout[k]
                    del iout[k]
                else:
                    merged.setdefault(k, [out[k]])
        for iout in merge:
            for k in merged.keys():
                merged[k].append(iout.get(k, out[k]))
        if not self.args.piecewise_param_values:
            xmax = out[f'{self.args.piecewise_param.title()}Max']
            self.args.piecewise_param_values = list(
                np.linspace(0, xmax, len(merge) + 2,
                            dtype=type(xmax))[1:-1])
        assert len(self.args.piecewise_param_values) == len(merge)
        generator = self.generator_class(**out)
        for k, v in merged.items():
            if k.endswith("XVals"):
                assert k.replace("XVals", "YVals") in merged
                continue
            elif k.endswith("YVals"):
                kx = k.replace("YVals", "XVals")
                assert k.endswith("YVals")
                # TODO: Fix this
                parent = generator.get(k, return_container=True,
                                       return_other='instance')
                xstep = [None] + [
                    parent.normalize(xx) for xx in
                    self.args.piecewise_param_values
                ]
                xout = []
                yout = []
                for i, (xmin, yvalues) in enumerate(zip(xstep, v)):
                    xvalues = merged[kx][i] if kx in merged else out[kx]
                    if i < (len(v) - 1):
                        xmax = xstep[i + 1]
                    else:
                        xmax = None
                    if xmin is None:
                        idx = (xvalues < xmax)
                    elif xmax is None:
                        idx = (xvalues >= xmin)
                    else:
                        idx = np.logical_and(
                            xvalues < xmax,
                            xvalues >= xmin
                        )
                    xout.append(xvalues[idx])
                    yout.append(yvalues[idx])
                xout = np.hstack(xout)
                yout = np.hstack(yout)
                out[k] = yout
                out[kx] = xout
            else:
                raise NotImplementedError(k)
        return out

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
            if self.args.piecewise_param:
                return self.generate_piecewise_merge()
            return self.generate_parameters()
        super(ParametrizeCropTask, self)._generate_output(name)


class LayoutTask(TaskBase):
    r"""Class for plotting the layout of a canopy."""

    _name = 'layout'
    _help = 'Plot the layout of a generated canopy'
    _output_info = {
        'layout': {
            'base_prefix': True,
            'ext': '.png',
            'description': 'a plot of the layout',
            'composite_param': ['canopy', 'periodic_canopy']
        },
    }
    _arguments = [
        (('--canopy', ), {
            'choices': ['all', 'single', 'tile', 'unique', 'virtual',
                        'virtual_single'],
            'default': 'unique',
            'help': 'Type of canopy to layout',
            'suffix_param': {
                'prefix': 'canopy', 'title': True, 'noteq': 'single',
            },
        }),
        arguments.DimensionArgumentDescription([
            (('--plot-length', '--row-length'), {
                'no_cli': True,
                'units': 'cm',
                'units_arg': 'mesh_units',
                'help': 'Length of plot rows forming canopy (in cm)',
            }),
            (('--plot-width', ), {
                'no_cli': True,
                'units': 'cm',
                'units_arg': 'mesh_units',
                'help': 'Width of plot forming canopy (in cm)',
            }),
        ], name='plot_dimensions'),
        arguments.DimensionArgumentDescription([
            (('--row-spacing', ), {
                'units': 'cm', 'default': 76.2,
                'units_arg': 'mesh_units',
                'help': 'Space between adjacent rows in plot (in cm)',
                'suffix_param': {'noteq': parse_quantity(76.2, 'cm')},
            }),
            (('--plant-spacing', '--col-spacing'), {
                'units': 'cm', 'default': 18.3,
                'units_arg': 'mesh_units',
                'help': 'Space between adjacent plants in rows (in cm)',
                'suffix_param': {'noteq': parse_quantity(18.3, 'cm')},
            }),
        ], name='plot_spacing', suffix_param={
            'cond': lambda x: (x.periodic_canopy or x.canopy != 'single'),
        }),
        arguments.DimensionArgumentDescription([
            (('--nrows', ), {
                'type': int, 'default': 4,
                'help': 'Number of rows to generate in plot',
                'suffix_param': {'noteq': 4},
            }),
            (('--ncols', ), {
                'type': int, 'default': 10,
                'help': 'Number of plants to generate in each row',
                'suffix_param': {'noteq': 10},
            }),
        ], name='plant_count', suffix_param={
            'cond': lambda x: (x.canopy != 'single'),
        }),
        arguments.ArgumentDescriptionSet([
            (('--periodic-canopy', ), {
                'nargs': '?', 'const': 'scene', 'default': False,
                'choices': [False, 'scene', 'plants', 'rays'],
                'help': (
                    'Make the canopy periodic for ray tracing so '
                    'that is infinitely wide'
                ),
                'suffix_param': {
                    'prefix': 'periodic', 'title': True,
                },
            }),
            (('--periodic-canopy-count', ), {
                'type': int,
                'help': (
                    'Number of times the canopy should be repeated in '
                    'each direction'
                ),
                'suffix_param': {},
            }),
        ], name='periodic_canopy_properties', suffix_param={
            'sep': '',
            'cond': lambda x: bool(x.periodic_canopy),
        }),
        arguments.CompositeArgumentDescription(
            'location',
            description=' that the sun should be modeled for',
            defaults={
                'location': 'Champaign',
            },
            suffix_param={'noteq': 'Champaign'},
        ),
        arguments.CompositeArgumentDescription(
            'time', description=' that the sun should be modeled for',
            ignore=['age', 'planting_date'],
            optional=True,
            suffix_param={},
        ),
        arguments.DimensionArgumentDescription([
            (('-x', '--x', '--row-offset'), {
                'units': 'cm', 'default': 0.0,
                'units_arg': 'mesh_units',
                'help': ('Starting position in the x direction '
                         '(perpendicular to rows)'),
            }),
            (('-y', '--y', '--plant-offset'), {
                'units': 'cm', 'default': 0.0,
                'units_arg': 'mesh_units',
                'help': ('Starting position in the y direction (along '
                         'rows)'),
            }),
        ], name='plot_offsets'),
        (('--plantid', ), {
            'type': int, 'default': 0,
            'help': 'Starting plant ID',
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
            'default': 0.0, 'units': 'meters',
            'units_arg': 'mesh_units',
            'help': ('Distance that the ground is above 0 along the '
                     '\"axis_up\" direction'),
        }),
        # TODO: Use mesh units in input
        (('--mesh-units', ), {
            'type': parse_units, 'default': units.Units('cm'),
            'help': 'Units that mesh should be output in',
        }),
        (('--real-plant-color', ), {
            'type': parse_color, 'default': 'red',
            'help': ('Color that should be used for the real plant when '
                     'a virtual canopy is created. '
                     'This should be a color string or 3 '
                     'comma separated RGB values expressed as floats in '
                     'the range [0, 1]'),
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

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        super(LayoutTask, cls).adjust_args_internal(args, **kwargs)
        args.plot_width = args.nrows * args.row_spacing
        args.plot_length = args.ncols * args.plant_spacing
        args.axis_cols = np.cross(args.axis_up, args.axis_rows)
        args.axis_east = np.cross(args.axis_north, args.axis_up)
        if args.periodic_canopy is True:
            args.periodic_canopy = 'scene'
        if args.canopy == 'single':
            args.nrows = 1
            args.ncols = 1
            args.plot_width = args.row_spacing
            args.plot_length = args.plant_spacing
        if args.periodic_canopy_count is None:
            if args.periodic_canopy:
                args.periodic_canopy_count = 2
            else:
                args.periodic_canopy_count = 0
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
        args.periodic_canopy_count_array = np.array([
            args.periodic_canopy_count,
            args.periodic_canopy_count,
            0,
        ], 'i4')
        args.virtual_canopy_count_array = np.ones((3, ), 'i4')
        args.virtual_period = np.array([
            args.row_spacing, args.plant_spacing, 0.0
        ], 'f4')
        args.virtual_direction = args.periodic_direction
        if args.canopy.startswith('virtual'):
            args.virtual_canopy_count_array[0] = args.nrows
            args.virtual_canopy_count_array[1] = args.ncols

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
        out = pos0 + units.QuantityArray(
            utils.get_periodic_shifts(
                self.args.virtual_period.astype('f4'),
                self.args.virtual_direction.astype('f4'),
                np.array([self.args.nrows, self.args.ncols, 0], 'i4'),
                include_origin=True,
                dont_reflect=True,
                dont_center=True,
            ).astype('f4'),
            self.args.plant_spacing.units,
        )
        return out

    @cached_property
    def nplants_virtual(self):
        r"""int: Number of plants in the virtual canopy buffer."""
        if self.args.canopy.startswith('virtual'):
            return (self.args.ncols * self.args.nrows) - 1
        return 0

    @cached_property
    def nplants_periodic(self):
        r"""int: Number of plants in the periodic canopy buffer."""
        # TODO: Allow multiple crop classes?
        if not self.args.periodic_canopy:
            return 0
        out = np.prod(2 * self.args.periodic_canopy_count_array + 1) - 1
        return self.nplants * out

    @cached_property
    def plant_positions_periodic(self):
        r"""np.ndarray: Locations of each plant in the periodic canopy
        buffer, in the order they are added to the scene."""
        # TODO: Allow all?
        if not self.args.periodic_canopy:
            return self.args.plant_spacing * np.zeros((0, 3), 'f4')
        pos_units0 = self.plant_positions.units
        pos0 = self.plant_positions
        shifts = units.QuantityArray(
            utils.get_periodic_shifts(
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
            'plants': utils.project_onto_ground(
                self.plant_positions,
                self.args.axis_rows, self.args.axis_cols,
            ),
            'periodic_plants': utils.project_onto_ground(
                self.plant_positions_periodic,
                self.args.axis_rows, self.args.axis_cols,
            ),
            'north': utils.project_onto_ground(
                self.args.axis_north,
                self.args.axis_rows, self.args.axis_cols, ray=True,
            ),
            'east': utils.project_onto_ground(
                self.args.axis_east,
                self.args.axis_rows, self.args.axis_cols, ray=True,
            ),
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
        if plantid == 0 and self.args.canopy.startswith('virtual'):
            return self.args.real_plant_color
        if self.isExteriorPlant(plantid):
            return self.args.exterior_plant_color
        return self.args.interior_plant_color

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
        raysun = utils.project_onto_ground(
            -self.get_solar_direction(time=time),
            self.args.axis_rows, self.args.axis_cols, ray=True,
        )
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
            'description': 'mesh',
            'upstream': ['parametrize', 'lpy_model'],
            'merge_all': 'all_combined',
            'composite_param': ['canopy'],
        },
        'geometryids': {
            'ext': '.csv',
            'base_output': 'generate',
            'base_suffix': True,
            'description': (
                'the IDs of the plant/component each face '
                'belongs to'
            ),
            'merge_all': 'all_combined',
        },
    }
    _external_tasks = {
        ParametrizeCropTask: {
            'include': [
                'crop', 'id', 'data', 'data_year',
                'debug_param', 'debug_param_prefix',
                'piecewise_param', 'piecewise_param_values',
                'piecewise_files', 'piecewise_ids',
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
        LayoutTask: {
            'include': [
                'canopy', 'plot_length', 'plot_width',
                'nrows', 'ncols', 'row_spacing', 'plant_spacing',
                'x', 'y', 'plantid', 'mesh_units',
            ],
            'modifications': {
                'canopy': {
                    'default': 'single',
                    'remove_choices': ['virtual', 'virtual_single'],
                    'suffix_param': {
                        'prefix': 'canopy', 'title': True,
                        'cond': lambda x: (
                            x.canopy not in ['single', 'virtual',
                                             'virtual_single']),
                    },
                },
                'plantid': {
                    'suffix_param': {'cond': lambda x: (x.plantid > 0)},
                },
                'plot_spacing': {
                    'suffix_param': {
                        'cond': lambda x: (
                            x.canopy not in ['single', 'virtual',
                                             'virtual_single']),
                    },
                },
                'plant_count': {
                    'suffix_param': {
                        'cond': lambda x: (
                            x.canopy not in ['single', 'virtual',
                                             'virtual_single']),
                    },
                },
            },
            'optional': True,
        },
    }
    _arguments = [
        arguments.CompositeArgumentDescription(
            'age', description=' to generate model for',
            ignore=['planting_date'],
            suffix_param={'noteq': 'maturity'},
        ),
        arguments.CompositeArgumentDescription(
            'color', description=(
                'that should be used for the generated plant. If '
                'a value of \'plantid\' is provided, colors will '
                'be used to identify individual plants by setting '
                'the blue channel to the plant ID.'
            ),
            optional=True,
            suffix_param={'noteq': None},
        ),
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
    ]

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
        if not cls.mesh_generated(args):
            return []
        if args.data:
            return utils.DataProcessor.from_file(args.data).ids
        elif args.data_year and not utils.DataProcessor._ignore_data:
            data = utils.DataProcessor.output_name(args.crop,
                                                   args.data_year)
            if os.path.isfile(data):
                return utils.DataProcessor.from_file(data).ids
        return ['default']

    @classmethod
    def base_param_class(cls, args):
        r"""Determine the base parameters for the provided args.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            dict: Base parameters.

        """
        years = utils.DataProcessor.available_years(args.crop)
        if (not years) or args.id == 'default':
            return {'id': 'default', 'data_year': None}
        ids = utils.DataProcessor.available_ids(args.crop, year=years[0])
        assert ids
        return {'id': ids[0], 'data_year': years[0]}

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
        return super(GenerateTask, cls)._read_output(args, name, fname)

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
        super(GenerateTask, cls)._write_output(args, name, fname, output)

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        args.save_all_plantids = False
        if args.verbose:
            args.verbose_lpy = True
        if args.plantid > 0 and not args.save_all_plantids:
            args.dont_write_generate = True
            args.dont_write_geometryids = True
        super(GenerateTask, cls).adjust_args_internal(args, **kwargs)
        if args.overwrite_generate:
            args.overwrite_geometryids = True

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

    @classmethod
    def shift_mesh(cls, args, mesh, x, y, plantid=None, geometryids=None):
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
        if geometryids is not None:
            xy = units.QuantityArray([x.value, y.value], x.units)
            geometryids['HEADER_JSON']['field_mins'] += xy
            geometryids['HEADER_JSON']['field_maxs'] += xy
            if plantid is not None:
                geometryids['plantids'] += plantid
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
            geometryids = []
            plantid = self.args.plantid
            assert plantid == 0
            for x in values:
                x['plantids'] += plantid
                geometryids.append(x)
                plantid += (self.args.nrows * self.args.ncols)
            geometryids = self.stack_geometryids(geometryids)
            return geometryids

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id not in ['all', 'all_combined']
        if name in ['generate', 'geometryids']:
            if self.args.canopy in ['single', 'virtual', 'virtual_single']:
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
            geometryids = copy.deepcopy(inst.get_output('geometryids'))
            mesh = self.shift_mesh(
                self.args, inst.get_output('generate'), x, y, plantid,
                geometryids=geometryids,
            )
            self.finalize_geometryids(geometryids)
            return (mesh, geometryids)
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
        face_areas = parse_quantity(mesh.areas, self.args.mesh_units**2)
        mesh_dict = utils.get_mesh_dict(mesh)
        vert_heights = (
            np.dot(mesh_dict['vertex'], self.args.axis_up)
            - self.args.ground_height.value
        )
        vert_ground = utils.project_onto_ground(
            mesh_dict['vertex'],
            self.args.axis_rows, self.args.axis_cols,
        )
        if len(mesh_dict['face']) == 0:
            face_heights = parse_quantity(
                np.zeros((0, ), dtype=vert_heights.dtype),
                self.args.mesh_units)
        else:
            face_heights = parse_quantity(
                vert_heights[mesh_dict['face']].mean(axis=1),
                self.args.mesh_units)
        for i, k in enumerate(component_order):
            for prev, count in components[k]:
                componentids[prev:(prev + count)] = i
        geometryids = {
            'HEADER_JSON': {
                'component_order': component_order,
                'field_mins': parse_quantity(
                    vert_ground.min(axis=0) if len(vert_ground)
                    else np.zeros((2, ), vert_ground.dtype),
                    self.args.mesh_units),
                'field_maxs': parse_quantity(
                    vert_ground.max(axis=0) if len(vert_ground)
                    else np.zeros((2, ), vert_ground.dtype),
                    self.args.mesh_units),
                'planting_density': 1.0 / (
                    self.args.plant_spacing * self.args.row_spacing),
            },
            'plantids': plantids, 'componentids': componentids,
            'areas': face_areas,
            'height': face_heights,
        }
        self.finalize_geometryids(geometryids)
        return (mesh, geometryids)

    @classmethod
    def finalize_geometryids(cls, dst):
        r"""Finalize the data in the JSON header, calculating derived
        values.

        Args:
            dst (dict): Geometry ID data to finalize.

        """
        header = dst['HEADER_JSON']
        field_area = utils.safe_op(
            np.prod, (
                header['field_maxs']
                - header['field_mins']
            ),
        )
        if 'Leaf' in header['component_order']:
            idx = (dst['componentids']
                   == header['component_order'].index('Leaf'))
            # TODO: Calculate leaf area directly from the generator?
            leaf_area = utils.safe_op(np.sum, dst['areas'][idx]) / 2
        else:
            leaf_area = units.Quantity(0.0, dst['areas'].units)
        planting_density = header['planting_density']
        LAI = leaf_area * planting_density
        header.update(
            field_area=field_area,
            leaf_area=leaf_area,
            LAI=LAI,
        )

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

    @classmethod
    def stack_geometryids(cls, geometryids):
        r"""Combine geometry IDs during field creation.

        Args:
            geometryids (list): Set of geometry IDs for individual
                plants/fields.

        Returns:
            dict: Geometry IDs.

        """
        out = {}
        order = geometryids[0]['HEADER_JSON']['component_order']
        for x in geometryids[1:]:
            if len(x['HEADER_JSON']['component_order']) > len(order):
                order = x['HEADER_JSON']['component_order']
        for x in geometryids:
            xorder = x['HEADER_JSON']['component_order']
            if xorder == order:
                continue
            for i, name in enumerate(xorder):
                assert name in order
                x['componentids'][x['componentids'] == i] = (
                    order.index(name) + len(order)
                )
            x['componentids'] -= len(order)
            x['HEADER_JSON']['component_order'] = order
        for k in geometryids[0].keys():
            if k == 'HEADER_JSON':
                out[k] = geometryids[0][k]
                assert all(
                    x['HEADER_JSON']['component_order']
                    == out[k]['component_order']
                    for x in geometryids[1:]
                )
            else:
                out[k] = np.hstack([x[k] for x in geometryids])
        out['HEADER_JSON']['field_mins'] = utils.safe_op(
            np.min, np.vstack([
                x['HEADER_JSON']['field_mins'] for x in geometryids
            ]), axis=0,
        )
        out['HEADER_JSON']['field_maxs'] = utils.safe_op(
            np.max, np.vstack([
                x['HEADER_JSON']['field_mins'] for x in geometryids
            ]), axis=0,
        )
        cls.finalize_geometryids(out)
        return out

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
            igeometryids = copy.deepcopy(inst.get_output('geometryids'))
            mesh = self.shift_mesh(
                self.args, inst.get_output('generate'), x, y, plantid,
                geometryids=igeometryids,
            )
            if components is None:
                components = igeometryids['HEADER_JSON']['component_order']
            geometryids.append(igeometryids)
            geometryids = self.stack_geometryids(geometryids)
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
        igeometryids = inst.get_output('geometryids')
        if components is None:
            components = igeometryids['HEADER_JSON']['component_order']
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
                    geometryids.append(inst.get_output('geometryids'))
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
                    igeometryids = copy.deepcopy(geometryids[0])
                    mesh.append(
                        self.shift_mesh(
                            self.args, mesh_single,
                            ix + posdev(), iy + posdev(),
                            plantid=plantid,
                            geometryids=igeometryids,
                        )
                    )
                    geometryids.append(igeometryids)
                    plantid += 1
        else:
            raise ValueError(
                f"Unsupported canopy type: {self.args.canopy}")
        geometryids = self.stack_geometryids(geometryids)
        return (mesh, geometryids)


__all__ = ["monocot", "maize", "dicot", "tomato"]
