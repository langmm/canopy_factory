import os
import copy
import numpy as np
import yggdrasil_rapidjson as rapidjson
import argparse
from collections import OrderedDict
from yggdrasil_rapidjson import units
from canopy_factory import utils, arguments
from canopy_factory.utils import (
    check_optional_dependency,
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

def _piecewise_properties_suffix(x):
    r"""Build a suffix describing configured piecewise parameters.

    Args:
        x (object): Object with ``piecewise_param``,
            ``piecewise_param_values``, ``piecewise_files``, and
            ``piecewise_ids`` attributes.

    Returns:
        str: Underscore-joined parameter and value/ID pairs, or an empty
            string if piecewise parameters are not configured.

    """
    if not (x.piecewise_param and x.piecewise_param_values
            and (x.piecewise_files or x.piecewise_ids)):
        return ''
    values = x.piecewise_param_values
    if x.piecewise_param in ["N", "Age"]:
        values = [int(v) for v in values]
    if x.piecewise_files:
        ids = [
            os.path.splitext(os.path.basename(f))[0]
            for f in x.piecewise_files
        ]
    else:
        ids = x.piecewise_ids
    out = [
        f'{x.piecewise_param}{v}_{k}'
        for v, k in zip(values, ids)
    ]
    out = '_'.join(out)
    return out


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
            'composite_param': ['id', 'data_year', 'iteration_file'],
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
                'suffix_param': {'cond': False},
            }),
            (('--piecewise-param-values', ), {
                'type': float, 'nargs': '+', 'action': 'extend',
                'help': (
                    'Independent variable values at which parameters sets '
                    'should be changed'
                ),
                'suffix_param': {'cond': False},
            }),
            (('--piecewise-files', ), {
                'type': str, 'nargs': '+', 'action': 'extend',
                'help': (
                    'Names of parameter files that should be used '
                    'conditionally in the order they should be used'
                ),
                'suffix_param': {'cond': False},
            }),
            (('--piecewise-ids', ), {
                'type': str, 'nargs': '+', 'action': 'extend',
                'help': (
                    'IDs for parameters that should be used '
                    'conditionally in the order they should be '
                    'used (if parameter files not provided)'
                ),
                'suffix_param': {'cond': False},
            }),
        ], name='piecewise_properties', suffix_param={
            'cond': lambda x: bool(x.piecewise_param),
            'outputs': ['parametrize'],
            'skip_outputs': ['lpy_model'],
            'value': _piecewise_properties_suffix,
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
    _piecewise_arguments = ['piecewise_param', 'piecewise_param_values',
                            'piecewise_files', 'piecewise_ids']

    @staticmethod
    def _on_registration(cls):
        r"""Register unit arguments and complete class registration.

        Args:
            cls (type): Class being registered.

        """
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
        if not self.is_root:
            self.replace_iterating_parameters()

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
        if (((isinstance(idstr, str) and idstr.startswith('all'))
             or isinstance(idstr, list))):
            idstr = None
        fdata = None
        if ((idstr is not None
             and idstr in utils.DataProcessor.available_ids(
                 args.crop, year=args.data_year,
                 existing_param=True))):
            fname = os.path.join(utils.cfg['directories']['param'],
                                 f'{args.crop}_{idstr}.json')
            param = cls._read_output(args, 'parametrize', fname)
            idstr = param['id']
            args.id = idstr
            for k in (['crop', 'data', 'data_year']
                      + cls._piecewise_arguments):
                if getattr(args, k):
                    assert getattr(args, k) == param[k]
                else:
                    setattr(args, k, param[k])
        if args.data:
            fdata = utils.DataProcessor.from_file(args.data)
            args.data_year = fdata.year
        elif args.data_year and not (
                (isinstance(args.data_year, str)
                 and args.data_year.startswith('all'))
                or isinstance(args.data_year, list)):
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
                npiecewise = None
                if args.piecewise_files:
                    npiecewise = len(args.piecewise_files)
                    args.piecewise_ids = None
                elif args.piecewise_ids:
                    npiecewise = len(args.piecewise_ids)
                else:
                    id_base = idstr.split('_')[0]
                    ids = utils.DataProcessor.available_ids(
                        args.crop, year=args.data_year)
                    args.piecewise_ids = [
                        x for x in ids
                        if x.startswith(id_base) and x != idstr
                    ]
                    npiecewise = len(args.piecewise_ids)
                if args.piecewise_param_values is None:
                    args_overwrite = {
                        k: None for k in cls._piecewise_arguments
                    }
                    args_overwrite['output_parametrize'] = None
                    base = cls.run_iteration_class(
                        args,
                        args_overwrite=args_overwrite,
                        output_name='parametrize',
                    )
                    xmax = base[f'{args.piecewise_param.title()}Max']
                    dtype = type(xmax)
                    if args.piecewise_param.title() in ['N', 'Age']:
                        dtype = np.int32
                    args.piecewise_param_values = list(np.linspace(
                        0, xmax, npiecewise + 2, dtype=dtype
                    )[1:-1])
            else:
                args.piecewise_ids = []
                args.piecewise_files = []
                args.piecewise_param_values = []

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
        elif parameters is None:
            return None
        elif name == 'mature':
            return parameters['AgeMature']
        elif f'Age{name.title()}' in parameters:
            return parameters[f'Age{name.title()}']
        else:
            raise NotImplementedError(f'Invalid age \"{name}\"')

    def get_ages(self):
        r"""Get various ages defining the generated crops development.

        Returns:
            dict: Mapping of age names to values.

        """
        out = {
            k: self.get_age(k)
            for k in TimeArgument._age_strings
        }
        return out

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
        out_sorted = OrderedDict()
        for k in sorted(out.keys()):
            out_sorted[k] = out[k]
        return out_sorted

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
        args_overwrite = {
            k: None for k in self._piecewise_arguments
        }
        args_overwrite['output_parametrize'] = None
        out = copy.deepcopy(
            self.run_iteration(
                args_overwrite=args_overwrite,
                output_name='parametrize',
            )
        )
        merge = []
        if self.args.piecewise_files:
            for x in self.args.piecewise_files:
                merge.append(self.read_output('parametrize', x))
        else:
            for iid in self.args.piecewise_ids:
                args_overwrite['id'] = iid
                merge.append(self.run_iteration(
                    args_overwrite=args_overwrite,
                    output_name='parametrize',
                ))
        merged = {}
        assert len(self.args.piecewise_param_values) == len(merge)
        for iout in merge:
            for k in list(iout.keys()):
                kequal = (iout[k] == out.get(k, None))
                if isinstance(kequal, np.ndarray):
                    kequal = kequal.all()
                if kequal:
                    del iout[k]
                elif k == 'id':
                    del iout[k]
                else:
                    merged.setdefault(k, [out[k]])
        for iout in merge:
            for k in merged.keys():
                merged[k].append(iout.get(k, out[k]))
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
        for k in self._piecewise_arguments:
            out[k] = getattr(self.args, k)
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
            'composite_param': ['canopy', 'periodic_canopy'],
            'is_plot': True,
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
                'suffix_param': {
                    'noteq': parse_quantity(18.3, 'cm'),
                    'cond': lambda x: (x.planting_density is None),
                },
            }),
        ], name='plot_spacing', suffix_param={
            'cond': lambda x: (x.periodic_canopy or x.canopy != 'single'),
        }),
        (('--planting-density', ), {
            'help': (
                'Planting density to use to overwride the plant '
                'spacing within rows for a given number of rows/columns'
            ),
            'units': 'hectare**-1',
            'suffix_param': {
                'cond': lambda x: (
                    x.planting_density is not None
                    and (x.periodic_canopy or x.canopy != 'single')),
                'conv': lambda x: int(x.value / 1000),
                'suffix': 'k',
            },
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
                'choices': [False, 'False', 'scene', 'plants', 'rays'],
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
        if args.planting_density is not None:
            args.plant_spacing = (
                1.0 / (args.planting_density * args.row_spacing)
            ).to(args.row_spacing.units)
        args.plot_width = args.nrows * args.row_spacing
        args.plot_length = args.ncols * args.plant_spacing
        args.axis_cols = np.cross(args.axis_up, args.axis_rows)
        args.axis_east = np.cross(args.axis_north, args.axis_up)
        # planting_density = (
        #     1.0 / (args.plant_spacing * args.row_spacing)).to(
        #         'hectare**-1')
        # print(f'PLANTING DENSITY {planting_density}')
        # 74k 89k 104k 119k 133k
        # 76.2cm 51cm
        # 71712.34743198083 by default
        if args.periodic_canopy == "False":
            args.periodic_canopy = False
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
        ], 'f8')
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
        ])
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
        pos0 = self.args.plant_spacing * np.zeros((1, 3), 'f8')
        out = pos0 + units.QuantityArray(
            utils.get_periodic_shifts(
                self.args.virtual_period,
                self.args.virtual_direction,
                np.array([self.args.nrows, self.args.ncols, 0], 'i4'),
                include_origin=True,
                dont_reflect=True,
                dont_center=True,
            ),
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
            return self.args.plant_spacing * np.zeros((0, 3), 'f8')
        pos_units0 = self.plant_positions.units
        pos0 = self.plant_positions
        shifts = units.QuantityArray(
            utils.get_periodic_shifts(
                self.args.periodic_period,
                self.args.periodic_direction,
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
        r"""Return the color that should be used to plot a plant.

        Args:
            plantid (int): ID of the plant to get the color for.

        Returns:
            str: Color for the plant.

        """
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
            ax.plot(pos[0].value, pos[1].value,
                    color=self._get_color(plantid),
                    marker='o')
            ax.annotate(str(plantid), xy=pos.value, textcoords='data')
            plantid += 1
        for pos in layout['periodic_plants']:
            ax.plot(pos[0].value, pos[1].value,
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


class FieldPositionGenerator:

    r"""Class for generating random plant positions within a field.

    Positions are drawn from a normal distribution seeded by the starting
    plant ID so that the same sequence of deviations is produced when
    plants are generated out of order.

    """
    def __init__(self, args):
        self.args = args
        self.generator = None
        self.current_plantid = None
        self.reset()

    def reset(self):
        r"""Reset the generator to the starting plant ID."""
        self.generator = np.random.default_rng(seed=self.args.plantid)
        self.current_plantid = self.args.plantid

    def _generate(self):
        r"""Generate the position deviation for the next plant.

        Returns:
            tuple: Row and column deviations as quantities, both zero for
                the first plant in the field.

        """
        if self.current_plantid == self.args.plantid:
            out = tuple([
                parse_quantity(0.0, self.args.plant_spacing.units)
                for _ in range(2)
            ])
        else:
            out = tuple([
                parse_quantity(
                    self.generator.normal(
                        0.0,
                        self.args.location_stddev
                        * self.args.plant_spacing.value
                    ),
                    self.args.plant_spacing.units
                )
                for _ in range(2)
            ])
        self.current_plantid += 1
        return out

    def __call__(self, plantid):
        assert plantid >= self.args.plantid
        if self.current_plantid > plantid:
            self.reset()
        while self.current_plantid < plantid:
            self._generate()
        assert self.current_plantid == plantid
        return self._generate()


class PlantMesh(utils.RegisteredClassBase):

    r"""Class for per-face plant geometry data and its backing mesh.

    Stores one row per mesh face giving the owning plant ID, component
    ID, and any per-component data such as ``Age``, along with the mesh
    file the faces reference. Face ranges are serialized sparsely using
    ``face_start`` and ``face_stop`` so that runs of faces sharing a
    component are stored only once.

    """
    _registry_key = 'file'
    _registry_name = 'plant'
    _names_header = [
        'component_order', 'mesh_file', 'field_info',
    ]
    _names_header_calc = [
        'field_mins', 'field_maxs',
        'planting_density', 'field_area', 'leaf_area', 'LAI',
    ]
    _names_optional = [
        'mesh_file', 'field_info',
    ]
    _names_body = [
        'plantids', 'componentids',
    ]
    _names_body_calc = [
        'areas', 'height', 'face_centers',
    ]
    _names_sparse = [
        'face_start', 'face_stop',
    ]
    _required_args = [
        'canopy', 'plantid', 'mesh_units', 'mesh_format',
        'ground_height', 'location_stddev',
        'axis_up', 'axis_rows', 'axis_cols',
        'plant_spacing', 'row_spacing',
        'nrows', 'ncols',
        'x', 'y',
    ]
    _required_args_match = [
        'mesh_units', 'mesh_format',
        'ground_height',
        'axis_up', 'axis_rows', 'axis_cols',
    ]

    def __init__(self, args, mesh=None, body=None, **kwargs):
        if body is None:
            body = {}
        self.args = argparse.Namespace()
        self._mesh = mesh
        self.body = body
        if isinstance(args, dict):
            for k in self._required_args:
                setattr(self.args, k, args[k])
        else:
            assert isinstance(args, argparse.Namespace)
            for k in self._required_args:
                setattr(self.args, k, getattr(args, k))
        missing = []
        for k in self._names_body + self._names_sparse:
            if ((k not in self.body
                 and k not in self._names_optional)):
                missing.append(k)
        for k in self._names_header:
            if k not in kwargs:
                setattr(self, k, None)
                if k in self._names_optional:
                    continue
                missing.append(k)
            else:
                setattr(self, k, kwargs.pop(k))
        if kwargs or missing:
            raise ValueError(f"Extra ({list(kwargs.keys())}) or "
                             f"missing ({missing}) arguments")
        if not (self._mesh is not None or self.mesh_file):
            raise ValueError("Either mesh or mesh_file must be provided")
        if not self.args.mesh_format:
            self.args.mesh_format = 'obj'
        if isinstance(self.args.mesh_units, str):
            self.args.mesh_units = units.Units(self.args.mesh_units)

    def __deepcopy__(self, memo):
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for k in self._names_header + ["args", "body"]:
            if k in ["mesh_file"]:
                clone.mesh_file = None
                continue
            setattr(clone, k, copy.deepcopy(getattr(self, k), memo))
        clone._mesh = type(self.mesh)(self.mesh)
        return clone

    @classmethod
    def from_components(cls, args, mesh, components,
                        mesh_file=None):
        r"""Construct a PlantMesh from per-component face ranges.

        Args:
            args (argparse.Namespace, dict): Arguments describing the plant
                and its field.
            mesh (object): Mesh containing all of the component faces.
            components (dict): Mapping from component name to a list of
                ``(first_face, nfaces, data)`` tuples, where ``data``
                optionally maps per-component data names to values.
            mesh_file (str, optional): Path to the file the mesh is stored
                in.

        Returns:
            PlantMesh: Instance built from the provided components.

        """
        component_order = sorted(list(components.keys()))
        ncomponents = sum(len(v) for v in components.values())
        body = {
            'face_start': np.empty((ncomponents, ), dtype=np.int64),
            'face_stop': np.empty((ncomponents, ), dtype=np.int64),
            'plantids': args.plantid * np.ones(
                (ncomponents, ), dtype=np.uint32),
            'componentids': np.empty((ncomponents, ), dtype=np.uint32),
        }
        icomp = 0
        for i, k in enumerate(component_order):
            for prev, count, data in components[k]:
                body['face_start'][icomp] = prev
                body['face_stop'][icomp] = prev + count - 1
                body['componentids'][icomp] = i
                if data:
                    for kdata, vdata in data.items():
                        if kdata not in body:
                            body[kdata] = np.zeros(
                                (ncomponents, ),
                                dtype=np.float64
                            )
                        if isinstance(vdata, units.Quantity):
                            vdata = vdata.value
                        body[kdata][icomp] = vdata
                icomp += 1
        idx = np.argsort(body['face_start'])
        body = {k: v[idx] for k, v in body.items()}
        out = cls(args, mesh=mesh, mesh_file=mesh_file,
                  component_order=component_order,
                  body=body)
        # if out.args.x.value > 0 or out.args.y.value > 0:
        #     out.shift(out.args.x, out.args.y)
        return out

    @classmethod
    def from_file(cls, fname):
        r"""Read a PlantMesh from a CSV file.

        Args:
            fname (str): Path to the CSV file to read. The mesh file stored
                in the header is resolved relative to this path.

        Returns:
            PlantMesh: Instance read from the file.

        """
        body = utils.read_csv(fname)
        header = body.pop("HEADER_JSON")
        args = header.pop("args")
        header['mesh_file'] = os.path.join(
            os.path.dirname(fname), header['mesh_file'])
        return cls(args, body=body, **header)

    @classmethod
    def expand_sparse(cls, body):
        r"""Expand sparse face ID ranges into one value per face.

        Args:
            body (dict): Columns including ``face_start`` and
                ``face_stop`` ranges.

        Returns:
            dict: Columns expanded to one value per face.

        """
        starts = np.asarray(body[cls._names_sparse[0]], dtype=np.int64)
        stops = np.asarray(body[cls._names_sparse[1]], dtype=np.int64)
        nfaces = int(stops[-1]) + 1 if len(stops) else 0
        out = {}
        for k, v in body.items():
            if k in cls._names_sparse:
                continue
            v = np.asarray(v)
            dense = np.zeros((nfaces, ), dtype=v.dtype)
            for i, (start, stop) in enumerate(zip(starts, stops)):
                dense[int(start):int(stop) + 1] = v[i]
            out[k] = dense
        return out

    def write(self, fname):
        r"""Write the mesh and its face data to files.

        The backing mesh is written next to ``fname`` using the configured
        mesh format if ``mesh_file`` is not already set.

        Args:
            fname (str): Path to the CSV file that the face data should be
                written to.

        """
        if not self.mesh_file:
            self.mesh_file = (
                os.path.splitext(fname)[0]
                + utils._inv_geom_ext[self.args.mesh_format]
            )
        assert os.path.dirname(fname) == os.path.dirname(self.mesh_file)
        utils.write_3D(self.mesh, self.mesh_file,
                       file_format=self.args.mesh_format)
        utils.write_csv(self.json, fname)

    @property
    def mesh(self):
        r"""object: Mesh, read from file if it is not already loaded."""
        if self._mesh is None:
            self._mesh = utils.read_3D(self.mesh_file)
        assert self._mesh.nface == self.nfaces
        return self._mesh

    @cached_property
    def nfaces(self):
        r"""int: Total number of faces in the mesh."""
        stops = self.body[self._names_sparse[1]]
        return int(stops[-1]) + 1 if len(stops) else 0

    @cached_property
    def facedata(self):
        r"""dict: Face data with sparse face ranges expanded."""
        return self.expand_sparse(self.body)

    @cached_property
    def face_properties(self):
        r"""list: Names of the per-face properties that are available."""
        out = [k for k in self.body.keys()
               if k not in self._names_sparse]
        out += self._names_body_calc
        return out

    def get_face_property(self, name):
        r"""Return a per-face property by name.

        Args:
            name (str): Name of the property to return.

        Returns:
            np.ndarray: Values of the property, one per face.

        """
        if name in self._names_body_calc:
            return getattr(self, name)
        return self.facedata[name]

    def face_property_mask(self, k, v):
        r"""Return a mask selecting faces with a given property value.

        Args:
            k (str): Name of the property to test.
            v: Value that selected faces should match.

        Returns:
            np.ndarray: Boolean mask over the faces.

        """
        assert k in self.face_properties
        return (self.get_face_property(k) == v)

    @property
    def plantids(self):
        r"""np.ndarray: Plant ID assigned to each face."""
        return self.facedata['plantids']

    @property
    def componentids(self):
        r"""np.ndarray: Component ID assigned to each face."""
        return self.facedata['componentids']

    @cached_property
    def mesh_dict(self):
        r"""dict: Mesh data as arrays of vertices and faces."""
        return utils.get_mesh_dict(self.mesh)

    @cached_property
    def face_centers(self):
        r"""np.ndarray: Centers of each face."""
        faces = self.mesh_dict['face']
        vertices = self.mesh_dict['vertex']
        return vertices[faces, :].mean(axis=1)

    @cached_property
    def areas(self):
        r"""units.Quantity: Area of each face in the mesh."""
        return parse_quantity(self.mesh.areas, self.args.mesh_units**2)

    @cached_property
    def vert_heights(self):
        r"""np.ndarray: Height of each vertex above the ground."""
        return (
            np.dot(self.mesh_dict['vertex'], self.args.axis_up)
            - self.args.ground_height.value
        )

    @cached_property
    def vert_ground(self):
        r"""np.ndarray: Ground-projected coordinates of each vertex."""
        return utils.project_onto_ground(
            self.mesh_dict['vertex'],
            self.args.axis_rows, self.args.axis_cols,
        )

    @cached_property
    def height(self):
        r"""units.Quantity: Mean height above ground of each face."""
        if len(self.mesh_dict['face']) == 0:
            face_heights = parse_quantity(
                np.zeros((0, ), dtype=self.vert_heights.dtype),
                self.args.mesh_units)
        else:
            face_heights = parse_quantity(
                self.vert_heights[self.mesh_dict['face']].mean(axis=1),
                self.args.mesh_units)
        return face_heights

    @cached_property
    def field_mins(self):
        r"""units.Quantity: Lower corner of the field bounding box."""
        return parse_quantity(
            self.vert_ground.min(axis=0) if len(self.vert_ground)
            else np.zeros((2, ), self.vert_ground.dtype),
            self.args.mesh_units
        )

    @cached_property
    def field_maxs(self):
        r"""units.Quantity: Upper corner of the field bounding box."""
        return parse_quantity(
            self.vert_ground.max(axis=0) if len(self.vert_ground)
            else np.zeros((2, ), self.vert_ground.dtype),
            self.args.mesh_units
        )

    @cached_property
    def unique_plantids(self):
        r"""list: Sorted unique plant IDs contained in the mesh."""
        return sorted(np.unique(self.plantids))

    @cached_property
    def planting_density(self):
        r"""units.Quantity: Planting density of the field."""
        return 1.0 / (self.args.plant_spacing * self.args.row_spacing)

    @cached_property
    def field_area(self):
        r"""units.Quantity: Area of the field bounding box."""
        return utils.safe_op(
            np.prod, (self.field_maxs - self.field_mins)
        )

    @cached_property
    def leaf_area(self):
        r"""units.Quantity: Total one-sided area of all leaves."""
        if 'Leaf' in self.component_order:
            idx = (self.componentids
                   == self.component_order.index('Leaf'))
            # TODO: Calculate leaf area directly from the generator?
            return utils.safe_op(np.sum, self.areas[idx]) / 2
        else:
            return units.Quantity(0.0, self.areas.units)

    @cached_property
    def LAI(self):
        r"""units.Quantity: Leaf area index of the field."""
        return self.leaf_area * self.planting_density

    @property
    def header(self):
        r"""dict: Header data written to the output file."""
        out = {k: getattr(self, k) for k in self._names_header}
        out['args'] = {
            k: getattr(self.args, k) for k in self._required_args
        }
        if 'mesh_file' in out:
            out['mesh_file'] = os.path.basename(out['mesh_file'])
        return out

    @property
    def json(self):
        r"""dict: Header and body data as a JSON-serializable dict."""
        out = {
            'HEADER_JSON': self.header
        }
        out.update(self.body)
        return out

    def set_units(self, new_units):
        r"""Scale the mesh into new units in place.

        Args:
    new_units (units.Units): Units to convert the mesh to.

        """
        if new_units == self.args.mesh_units:
            return
        self._mesh = utils.scale_mesh(
            self.mesh, 1.0,
            from_units=self.args.mesh_units,
            to_units=new_units
        )
        self.args.mesh_units = new_units
        self.clear_cached_properties()

    def in_units(self, new_units):
        r"""Return a copy of the mesh converted to new units.

        Args:
    new_units (units.Units): Units to convert the mesh to.

        Returns:
            PlantMesh: Copy of the mesh in the requested units.

        """
        out = copy.deepcopy(self)
        out.set_units(new_units)
        return out

    def add_colors(self, *args, **kwargs):
        r"""Add colors to the underlying mesh.

        Args:
            *args: Positional arguments passed to the mesh.
            **kwargs: Keyword arguments passed to the mesh.

        """
        self.mesh.add_colors(*args, **kwargs)

    def get_plant(self, plantid, mesh_file=None):
        r"""Extract a single plant from the mesh.

        Args:
            plantid (int): ID of the plant to extract.
            mesh_file (str, optional): Path to the file that the plant mesh
                is stored in.

        Returns:
            PlantMesh: Instance containing only the requested plant.

        """
        compmask = (self.body['plantids'] == plantid)
        kws = {k: getattr(self, k) for k in self._names_header
               if k not in ['mesh_file', 'field_info']}
        body = {}
        for k, v in self.body.items():
            body[k] = v[compmask].copy()
        first_face = body['face_start'].min()
        last_face = body['face_stop'].max()
        body['face_start'] -= first_face
        body['face_stop'] -= first_face
        facemask = slice(first_face, last_face + 1)
        # facemask = (self.plantids == plantid)
        mesh = utils.select_faces(self.mesh, facemask)
        args = copy.deepcopy(self.args)
        args.ncol = 1
        args.nrow = 1
        args.plantid = plantid
        return type(self)(args, mesh=mesh, body=body, **kws)

    def append(self, other):
        r"""Append the faces of one or more other meshes in place.

        The other mesh's component IDs are remapped to this mesh's
        component order and its face ranges are offset so that they
        follow the existing faces.

        Args:
            other (PlantMesh, list): Mesh or list of meshes to append.

        """
        if isinstance(other, list):
            for x in other:
                self.append(x)
            return
        for k in self._required_args_match:
            v_self = getattr(self.args, k)
            v_other = getattr(other.args, k)
            if isinstance(v_self, np.ndarray):
                assert (v_other == v_self).all()
            else:
                try:
                    assert v_other == v_self
                except AssertionError:
                    # TODO: This can be removed once fully debugged
                    print("CONFLICTING", k, v_self, v_other)
                    import pdb
                    pdb.set_trace()
                    raise
        nfaces_prev = self.nfaces
        other_componentids = other.body['componentids'].copy()
        if other.component_order != self.component_order:
            for i, name in enumerate(other.component_order):
                assert name in self.component_order
                other_componentids[other_componentids == i] = (
                    self.component_order.index(name)
                    + len(self.component_order)
                )
            other_componentids -= len(self.component_order)
        assert (
            list(self.body.keys())
            == list(other.body.keys())
        )
        for k, v in other.body.items():
            if k == "componentids":
                self.body[k] = np.hstack([
                    self.body[k], other_componentids])
            elif k in self._names_sparse:
                self.body[k] = np.hstack([self.body[k], v + nfaces_prev])
            else:
                self.body[k] = np.hstack([self.body[k], v])
        self.mesh.append(other.mesh)
        self.clear_cached_properties()

    @cached_property
    def nplants(self):
        r"""int: Total number of plants across all combined fields."""
        out = (self.args.ncols * self.args.nrows)
        if not self.field_info:
            return out
        for args in self.field_info:
            out += (args.ncols * args.nrows)
        return out

    def append_plant(self, other, **kwargs):
        r"""Append one or more plants at their next field position.

        Args:
            other (PlantMesh, list): Plant mesh or list of plant meshes to
                append.
            **kwargs: Additional keyword arguments are passed to
                :meth:`plantpos`.

        """
        if isinstance(other, list):
            for x in other:
                self.append_plant(x, **kwargs)
            return
        plantid = max(self.plantids) + 1
        assert plantid < self.nplants
        assert len(other.unique_plantids) == 1
        x, y = self.plantpos(plantid, **kwargs)
        self.append(
            other.shifted(x, y, plantid=(plantid - min(other.plantids))),
        )

    @classmethod
    def stack_plants(cls, meshes, nrows, ncols):
        r"""Combine plant meshes into a field of the given size.

        The first mesh is used as the base and the remaining meshes are
        appended at successive plant positions.

        Args:
            meshes (list): Plant meshes to stack.
            nrows (int): Number of rows in the resulting field.
            ncols (int): Number of columns in the resulting field.

        Returns:
            PlantMesh: Field containing all of the provided plants.

        """
        out = copy.deepcopy(meshes[0])
        out.args.nrows = nrows
        out.args.ncols = ncols
        generator = FieldPositionGenerator(out.args)
        for x in meshes[1:]:
            out.append_plant(x, generator=generator)
        return out

    @classmethod
    def stack(cls, meshes):
        r"""Combine meshes by appending their faces.

        Args:
            meshes (list): Meshes to combine, used in order.

        Returns:
            PlantMesh: Combined mesh.

        """
        out = copy.deepcopy(meshes[0])
        for x in meshes[1:]:
            out.append(x)
        return out

    @classmethod
    def field_offset(cls, args, row_spacing=None):
        r"""Return the offset to the next field along the row axis.

        Args:
            args (argparse.Namespace): Arguments for the current field.
            row_spacing (units.Quantity, optional): Row spacing of the
                current field. Defaults to ``args.row_spacing``.

        Returns:
            units.Quantity: Row-axis offset for the next field.

        """
        if row_spacing is None:
            row_spacing = args.row_spacing
        return args.x + args.row_spacing * (args.nrows + 2)

    @cached_property
    def field_edge(self):
        r"""units.Quantity: Row-axis edge of all combined fields."""
        out = self.field_offset(self.args)
        if not self.field_info:
            return out
        for args in self.field_info:
            out += self.field_offset(args)
        return out

    def append_field(self, other):
        r"""Append one or more fields next to the current field.

        Args:
            other (PlantMesh, list): Field mesh or list of field meshes to
                append.

        """
        if isinstance(other, list):
            for x in other:
                self.append_field(x)
            return
        x = self.field_edge
        y = 0.0 * self.args.plant_spacing
        plantid = max(self.plantids) + 1
        self.append(
            other.shifted(x, y, plantid=(plantid - min(other.plantids))),
        )
        if self.field_info is None:
            self.field_info = []
        self.field_info.append(copy.deepcopy(other.args))

    @classmethod
    def stack_fields(cls, meshes):
        r"""Combine meshes by appending them as adjacent fields.

        Args:
            meshes (list): Field meshes to combine, used in order.

        Returns:
            PlantMesh: Combined field mesh.

        """
        out = copy.deepcopy(meshes[0])
        for x in meshes[1:]:
            out.append_field(x)
        return out

    def shift(self, x, y, plantid=None):
        r"""Shift the mesh and its data in place.

        Args:
            x (units.Quantity): Distance to shift along the row axis.
            y (units.Quantity): Distance to shift along the column axis.
                plantid (int, optional): Amount to add to every plant ID.
                If not provided, the plant IDs are left unchanged.

        """
        xo = x.to(self.args.mesh_units).value
        yo = y.to(self.args.mesh_units).value
        if plantid is not None:
            self.body['plantids'] += plantid
        self._mesh = utils.shift_mesh(
            self.mesh, xo, yo, plantid=plantid,
            axis_up=self.args.axis_up,
            axis_x=self.args.axis_rows
        )
        self.args.x += x
        self.args.y += y
        if plantid is not None:
            self.args.plantid += plantid
        self.clear_cached_properties()

    def shifted(self, *args, **kwargs):
        r"""Return a copy of the mesh shifted by the given amount.

        Args:
            *args: Positional arguments are passed to :meth:`shift`.
            **kwargs: Keyword arguments are passed to :meth:`shift`.

        Returns:
            PlantMesh: Shifted copy of the mesh.

        """
        out = copy.deepcopy(self)
        out.shift(*args, **kwargs)
        return out

    def get_generator(self, args=None):
        r"""Return a position generator for the mesh's field.

        Args:
            args (argparse.Namespace, optional): Arguments to seed the
                generator with. Defaults to ``self.args``.

        Returns:
            FieldPositionGenerator: Generator for plant positions.

        """
        if args is None:
            args = self.args
        return FieldPositionGenerator(args)

    def plantpos(self, plantid, args=None, generator=None,
                 row_spacing=None, plant_spacing=None):
        r"""Return the row and column position of a plant.

        Args:
            plantid (int): ID of the plant to locate.
            args (argparse.Namespace, optional): Arguments describing the
                field. Defaults to ``self.args``.
            generator (FieldPositionGenerator, optional): Generator used
                to produce the random position deviations.
            row_spacing (units.Quantity, optional): Override for the row
                spacing.
            plant_spacing (units.Quantity, optional): Override for the
                plant spacing.

        Returns:
            tuple: Row and column position of the plant.

        """
        if args is None:
            args = self.args
        field_offset = 0.0 * args.row_spacing
        if ((plantid < max(self.plantids)
             and plantid > (args.nrows * args.ncols))):
            assert self.field_info
            for field_args in self.field_info:
                if plantid < (args.nrows * args.ncols):
                    break
                field_offset += self.field_offset(
                    args, row_spacing=row_spacing)
                plantid -= args.nrows * args.ncols
                args = field_args
        if row_spacing is not None or plant_spacing is not None:
            args = copy.deepcopy(args)
            if row_spacing is not None:
                args.row_spacing = row_spacing
            if plant_spacing is not None:
                args.plant_spacing = plant_spacing
        if generator is not None and generator.args != args:
            generator = None
        if generator is None:
            generator = FieldPositionGenerator(args)
        i = int(np.floor(plantid / args.ncols))
        j = plantid - (i * args.ncols)
        posdev = generator(plantid)
        out = (
            field_offset + i * args.row_spacing + posdev[0],
            j * args.plant_spacing + posdev[1]
        )
        return out


class GenerateTask(TaskBase):
    r"""Class for generating 3D canopies."""

    _name = 'generate'
    _help = 'Generate a canopy mesh'
    _output_info = {
        'generate': {
            'ext': '.csv',
            'description': 'mesh',
            'upstream': ['parametrize', 'lpy_model'],
            'merge_all': 'all_combined',
            'composite_param': ['canopy'],
            'force_reload': True,
            'sidecar_files': {
                'mesh': lambda x: os.path.splitext(x)[0] + '.obj'
            },
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
                'planting_density',
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
            suffix_param={'cond': True},
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
        (('--nproc', ), {
            'type': int, 'default': 1,
            'help': (
                'Number of processes to split the iterations'
                ' between'
            ),
            'suffix_param': {'cond': False},
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
        return super().all_ids_class(args)

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
        if args.age.string not in fname:
            print(args.age.string, fname)
            import pdb
            pdb.set_trace()
            raise AssertionError('Reading the wrong file')
        if name == "generate":
            # TODO: Ensure that args match?
            return PlantMesh.from_file(fname)
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
        if args.id == 'all' or isinstance(args.id, list):
            return
        if name == "generate":
            assert isinstance(output, PlantMesh)
            output.write(fname)
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
        if args.verbose:
            args.verbose_lpy = True
        super(GenerateTask, cls).adjust_args_internal(args, **kwargs)
        if ((args.output_generate is True
             and args.dont_write_generate is None)):
            args.dont_write_generate = False

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
        if name != 'generate':
            return super(GenerateTask, self)._merge_output(
                name, output, merged_param)
        assert self.args.id == 'all_combined'
        return PlantMesh.stack_fields(list(output.values()))

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id not in ['all', 'all_combined']
        assert not isinstance(self.args.id, list)
        if name == 'generate':
            if self.args.canopy in ['single', 'virtual', 'virtual_single']:
                return self._generate_single_plant()
            else:
                out = self._generate_field()
                if self.args.dont_write_generate is None:
                    self.args.output_generate.args['dont_write'] = True
                    assert not self.output_enabled('generate',
                                                   for_write=True)
                return out
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
        check_optional_dependency("openalea")
        from openalea.lpy import Lsystem
        x = self.args.x
        y = self.args.y
        plantid = self.args.plantid
        self.log(f'generate_single_plant: {x}, {y}, {plantid}')
        if (x.value > 0) or (y.value > 0):
            inst = self.run_iteration(
                args_overwrite={'x': 0.0, 'y': 0.0},
            )
            return inst.get_output('generate').shifted(x, y)
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
        derivation_length = 2 * int(age / self.parameters['AgeInc'])
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
        geom_kws = {}
        if self.args.color.string == 'plantids':
            geom_kws["color"] = [0, 255, plantid]
        elif self.args.color.string == 'componentids':
            raise NotImplementedError
        elif self.args.color.value is not None:
            geom_kws["color"] = self.args.color.value
        lpy_model = self.output_file('lpy_model')
        if not os.path.isfile(lpy_model):
            self.get_output('lpy_model')
            assert os.path.isfile(lpy_model)
        lsys = Lsystem(lpy_model, lpy_param)
        tree = lsys.derive()
        generator = lsys.context().globals()['generator']
        scene = lsys.sceneInterpretation(tree)
        lsys_units = generator.unit_system
        components = {}
        if not self.args.mesh_format:
            self.args.mesh_format = 'obj'
        raw_mesh = utils.scene2geom(
            scene, cls=self.args.mesh_format,
            axis_up=self.args.axis_up,
            axis_x=self.args.axis_rows,
            components=components,
            plant_parameters=generator,
            **geom_kws
        )
        raw_mesh = utils.scale_mesh(
            raw_mesh, 1.0,
            from_units=lsys_units.length,
            to_units=self.args.mesh_units,
        )
        if x.value > 0 or y.value > 0:
            raw_mesh = utils.shift_mesh(
                raw_mesh,
                x.to(self.args.mesh_units).value,
                y.to(self.args.mesh_units).value,
                axis_up=self.args.axis_up,
                axis_x=self.args.axis_rows,
            )
        return PlantMesh.from_components(self.args, raw_mesh, components)

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
        # self.get_parameters()
        x = self.args.x
        y = self.args.y
        plantid = self.args.plantid
        self.log(f'generate_field: {x}, {y}, {plantid}')
        generator = np.random.default_rng(seed=plantid)

        def posdev():
            r"""Return a random position deviation.

            Returns:
                units.Quantity: Offset drawn from a normal distribution.

            """
            return parse_quantity(generator.normal(
                0.0,
                self.args.location_stddev * self.args.plant_spacing.value
                ), self.args.plant_spacing.units
            )

        # First plant
        args_overwrite = {
            'x': 0.0, 'y': 0.0, 'plantid': plantid,
            'canopy': 'single', 'nrows': 1, 'ncols': 1,
            'output_parametrize': None,
            'output_generate': None,
        }
        inst = self.run_iteration(args_overwrite=args_overwrite)
        # TODO: Prime this in a systemic way
        # inst.get_output('parametrize')
        single_mesh = inst.get_output('generate').shifted(x, y)
        mesh = copy.deepcopy(single_mesh)
        mesh.args.ncols = self.args.ncols
        mesh.args.nrows = self.args.nrows
        mesh.args.canopy = self.args.canopy
        plantid += 1

        # Remainder of field
        if self.args.canopy == 'unique':
            kwargs_pool = []
            shifts_pool = []
            for i in range(self.args.nrows):
                ix = i * self.args.row_spacing
                for j in range(self.args.ncols):
                    iy = j * self.args.plant_spacing
                    if i == 0 and j == 0:
                        continue
                    ikwargs = dict(
                        args_overwrite,
                        plantid=plantid,
                    )
                    ishifts = (x + ix + posdev(), y + iy + posdev())
                    kwargs_pool.append(ikwargs)
                    shifts_pool.append(ishifts)
                    # The performance of the single plant generation
                    # needs improved
                    # imesh = self.run_iteration(
                    #     args_overwrite=ikwargs,
                    #     output_name="generate",
                    #     copy_outputs_from=inst,
                    # )
                    # mesh.append(imesh.shifted(x + ix + posdev(),
                    #                           y + iy + posdev()))
                    plantid += 1
            results = self.run_iterations(
                kwargs_pool, nproc=self.args.nproc,
                output_name="filename_generate",
                # copy_outputs_from=inst,
            )
            for imeshfile, ishifts in zip(results, shifts_pool):
                self.log(f"Appending {imeshfile}")
                imesh = self.read_output('generate', imeshfile)
                mesh.append(imesh.shifted(*ishifts))
        elif self.args.canopy == 'tile':
            for i in range(self.args.nrows):
                ix = i * self.args.row_spacing
                for j in range(self.args.ncols):
                    iy = j * self.args.plant_spacing
                    if i == 0 and j == 0:
                        continue
                    mesh.append(
                        single_mesh.shifted(
                            ix + posdev(), iy + posdev(),
                            plantid=plantid,
                        )
                    )
                    plantid += 1
        else:
            raise ValueError(
                f"Unsupported canopy type: {self.args.canopy}")
        return mesh


__all__ = ["monocot", "maize", "dicot", "tomato"]
