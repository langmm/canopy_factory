import pprint
import pdb
import copy
import functools
import numpy as np
import pandas as pd
import scipy
import uuid
import contextlib
import openalea.plantgl.all as pgl
import openalea.plantgl.math as pglmath
from openalea.plantgl.math import Vector2, Vector3, Vector4
from openalea.plantgl.scenegraph import (
    NurbsCurve2D, NurbsCurve, NurbsPatch)
import yggdrasil_rapidjson as rapidjson
from canopy_factory.cli import SubparserBase
from canopy_factory.utils import (
    RegisteredMetaClass, get_class_registry,
    NoDefault, jsonschema2argument, format_list_for_help, UnitSet,
    cached_property, DataProcessor,
    DictWrapper, DictSet, PrefixedDict, SimpleWrapper, temporary_prefix,
)


############################################################
# LPy parametrization class
#  - 'age' indicates the time since germination
#  - 'n' indicates the phytomer count
############################################################


class PlantParameterBase(SubparserBase):
    r"""Base class for managing architecture parameters."""

    _registry_key = 'plant_parameter'
    _properties = {}
    _defaults = {}
    _property_dependencies = {}
    _property_dependencies_defaults = {}
    _unit_dimension = None

    @staticmethod
    def _add_parameter_arguments(cls, dst, name=None,
                                 existing=None, **kws):
        if existing is None:
            existing = [x[0][0] for x in dst._arguments]
        if name is None:
            name = ''
        for k, v in cls._properties.items():
            cls._add_parameter_argument(cls, dst, name, existing,
                                        k, v, **kws)

    @staticmethod
    def _add_parameter_argument(cls, dst, name, existing, k, v,
                                parents=None, dependencies=None, **kws):
        if existing is None:
            existing = [x[0][0] for x in dst._arguments]
        if name is None:
            name = ''
        kname = name if name.endswith(k) else name + k
        if isinstance(v, type):
            parents = [] if parents is None else copy.deepcopy(parents)
            parents.append((name, cls))
            dependencies = cls._get_property_dependencies(
                k, prefix=name, existing=dependencies,
            )
            v._add_parameter_arguments(v, dst, name=kname,
                                       parents=parents,
                                       existing=existing,
                                       dependencies=dependencies,
                                       **kws)
            return
        new_arg = cls._property2argument(cls, dst, name, k, v,
                                         parents=parents,
                                         dependencies=dependencies,
                                         **kws)
        if new_arg[0][0] not in existing:
            dst._arguments.append(new_arg)
            existing.append(new_arg[0][0])

    @staticmethod
    def _property2argument(cls, dst, name, k, json,
                           dependencies=None, parents=None, **kws):
        kname = name if name.endswith(k) else name + k
        kwargs = jsonschema2argument(json, no_defaults=True)
        kname_arg = kname.replace('_', '-')
        # if kname in dst._defaults:
        #     kwargs['default'] = dst._defaults[kname]
        # else:
        #     kwargs.pop('default', None)
        if 'help' not in kwargs and cls._help:
            kwargs['help'] = cls._help
        kwargs.update(**kws)
        assert 'default' not in kwargs
        # kwargs.pop('default', None)
        dependencies = cls._get_property_dependencies(
            k, prefix=name, existing=dependencies,
        )
        if 'help' in kwargs:
            while '{' in kwargs['help']:
                kwargs['help'] = cls._format_help(
                    cls, dst, kwargs['help'], name, k, parents=parents
                )
            if dependencies[-1]:
                provided = []
                not_provided = []
                conditions = []
                for kdep, vdep in dependencies[-1].items():
                    if vdep is True:
                        provided.append(kdep)
                    elif vdep is False:
                        not_provided.append(kdep)
                    else:
                        assert isinstance(vdep, list)
                        conditions.append(f'{kdep} one of {vdep}')
                if provided:
                    verb = 'is' if len(provided) == 1 else 'are'
                    conditions.append(f'{provided} {verb} provided')
                if not_provided:
                    verb = 'is' if len(not_provided) == 1 else 'are'
                    conditions.append(f'{not_provided} {verb} not provided')
                conditions = format_list_for_help(conditions)
                kwargs['help'] += f' [Only valid if {conditions}]'
        if kname_arg != kname and 'dest' not in kwargs:
            kwargs['dest'] = kname
        if any(dependencies):
            kwargs['dependencies'] = dependencies
        return ((f'--{kname_arg}', ), kwargs)

    @staticmethod
    def _format_help(cls, dst, msg, name, k, parents=None):
        kname = name if name.endswith(k) else name + k
        fmtkws = {
            'class_name': cls._name,
            'class_help': cls._help,
            'parent': name,
            'Component': dst._property2component(kname),
        }
        if parents:
            fmtkws['parents'] = [x[0] for x in parents[::-1]]
        if cls._name is not None:
            for k, v in list(fmtkws.items()):
                if v is None:
                    fmtkws.pop(k)
        if isinstance(fmtkws.get('Component', None), str):
            fmtkws['component'] = fmtkws['Component'].lower()
        try:
            return msg.format(**fmtkws)
        except KeyError:
            print(cls, kname)
            raise

    @classmethod
    def _get_required_properties(cls, name=None):
        if name is None:
            out = []
            for k in cls._required:
                out += cls._get_required_properties(k)
            return out
        if isinstance(cls._properties[name], type):
            return [f'{name}{k}' for k in
                    cls._properties[name]._get_required_properties()]
        return [name]

    @classmethod
    def _get_property_dependencies(cls, name, prefix='', existing=None,
                                   base=None):
        out = [] if existing is None else copy.deepcopy(existing)
        out.append({} if base is None else copy.deepcopy(base))
        if name not in cls._property_dependencies:
            return out
        for k, v in cls._property_dependencies[name].items():
            reqprops = cls._get_required_properties(k)
            for x in reqprops:
                subx = x if prefix is None else f'{prefix}{x}'
                out[-1][subx] = v
        return out


def schema2parameter(name, schema):
    r"""Convert a schema into a parameter class.

    Args:
        name (str): Name of the schema the property belongs to.
        schema (dict): JSON schema.

    Returns:
        type: Specialized plant parameter class.

    """
    spec_name = f"{name}_{schema['type']}"
    kws = {}
    if 'default' in schema:
        kws['defaults'] = {'': schema['default']}
    if 'description' in schema:
        kws['_help'] = schema['description']
    cls = get_class_registry().get('plant_parameter', schema['type'])
    if kws:
        cls = cls.specialize(spec_name, **kws)
    return cls


def DelayedPlantParameter(name, **kwargs):
    r"""Class factory for creating a plant parameter class on
    registration.

    Args:
        name (str): Name of the delayed plant parameter class.
        **kwargs: Additional keyword arguments are passed to the
            specialize method for the class.

    Returns:
       type: Dummy plant parameter class that will access a registered
           plant parameter class when it is used to create an instance.

    """

    spec_name = kwargs.pop('specialize_name', None)

    def parameter_class():
        cls = get_class_registry().get('plant_parameter', name)
        if spec_name is not None:
            cls = cls.specialize(spec_name, **kwargs)
        return cls

    class DelayedPlantParameterMeta(type):

        def __call__(self, *args, **kwargs):
            return parameter_class()(*args, **kwargs)

        def __getattr__(self, k):
            return getattr(parameter_class(), k)

    class _DelayedPlantParameter(object, metaclass=DelayedPlantParameterMeta):
        pass

    return _DelayedPlantParameter


class ParameterValues(DictSet):
    r"""Class for managing a set of finalized parameter values.

    Args:
        instance (PlantParameterBase): Instance responsible for
            generating a parameter from the member parameters.

    """

    def __init__(self, instance):
        self.instance = instance
        self.index = instance.cache.index
        self.cache = PrefixedDict(instance.cache,
                                  prefix=instance.fullname,
                                  immutable=True)
        self.simple = PrefixedDict({})
        self.nested = PrefixedDict({})
        self.local = DictSet([self.simple, self.nested])
        members = [
            self.instance._constants,
            self.local,
        ]
        super(ParameterValues, self).__init__(members)

    @property
    def dest(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        return self.simple

    def clear(self):
        r"""Clear the set parameters."""
        self.simple.clear()
        self.nested.clear()

    def __setitem__(self, k, v):
        if isinstance(v, PlantParameterBase):
            self.nested[k] = v
        else:
            self.simple[k] = v

    @property
    def current_param(self):
        r"""str: Current parameter being generated."""
        return self.instance.current_param

    def get(self, k, default=NoDefault, return_other=None, **kwargs):
        r"""Get a parameter value.

        Args:
            k (str): Name of parameter to return, without any prefixes.
            default (object, optional): Default to return if the
                parameter is not set.
            return_other (str, optional): Name of what should be
                returned instead of the parameter value.
            **kwargs: Additional keyword arguments are passed to any
                nested calls to get or generate.

        Returns:
            object: Parameter value.

        """
        if k in ParameterIndex.parameter_names:
            v = self.index[k]
            if v is None:
                if default is NoDefault:
                    raise KeyError(f'Null index parameter \"{k}\"')
                v = default
            return v
        v = NoDefault
        if return_other not in [None, 'instance']:
            v = self.get(k, default=default, return_other='instance',
                         **kwargs)
            if isinstance(v, PlantParameterBase):
                v = getattr(v, return_other)
            return v
        try:
            if k == '' and self.current_param != self.instance.fullname:
                v = self.instance
            else:
                v = self[k]
        except KeyError:
            if v is NoDefault and k in self.instance.component_parameters:
                try:
                    v = self.instance.component.get(
                        k, return_other=return_other, **kwargs)
                except KeyError:
                    pass
            if v is NoDefault and k in self.instance.external_parameters:
                if k in self.instance.component.parameters:
                    src = self.instance.component
                else:
                    src = self.instance.root
                try:
                    v = src.get(k, return_other='instance')
                except KeyError:
                    pass
            if v is NoDefault:
                for kk, vv in self.nested.items():
                    if vv.prefixes(k):
                        v = vv.get(
                            vv.remove_prefix(k), return_other='instance')
                        break
            if v is NoDefault:
                if default is NoDefault:
                    raise KeyError(k)
                v = default
        if isinstance(v, PlantParameterBase) and return_other != 'instance':
            if not v.initialized:
                self.error(KeyError, f"Parameter not initialized \"{k}\"")
            v = v.generate(**kwargs)
        elif kwargs:
            raise AssertionError(f"Kwargs unused: {kwargs}")
        return v


class ParameterDict(DictSet):
    r"""Class for managing a set of parameters.

    Args:
        provided (dict, optional): Set of user provided parameters.

    """

    def __init__(self, provided=None, **kwargs):
        self.provided = PrefixedDict(provided, immutable=True, **kwargs)
        self.previous = PrefixedDict(DictSet([]), immutable=True, **kwargs)
        self.defaults = PrefixedDict(DictSet([]), immutable=True, **kwargs)
        self.component = None
        self._dest = PrefixedDict({}, **kwargs)
        members = [
            self.provided, self.previous, self._dest, self.defaults,
        ]
        super(ParameterDict, self).__init__(members, **kwargs)

    @property
    def dest(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        return self._dest

    @property
    def prefix(self):
        r"""str: Current prefix."""
        return self.dest.prefix

    @contextlib.contextmanager
    def temporary_source_prefix(self, prefix, **kwargs):
        r"""Temporarily create a context where the source dictionaries
        have a different prefix than the destination dictionary.

        Args:
            prefix (str): Temporary prefix to set for the sources.
            **kwargs: Additional keyword arguments are passed to
                temporary_prefix for each source dictionary.

        Yields:
            ParameterDict: Self with temporary prefix applied.

        """
        with contextlib.ExitStack() as stack:
            for x in [self.provided, self.previous, self.defaults]:
                stack.enter_context(x.temporary_prefix(prefix, **kwargs))
            yield self

    @contextlib.contextmanager
    def updating(self, cls):
        r"""Context manager for updating the parameters for a class.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated. Default parameters will be added to the current
                set based on the existing parameters and the class's
                prefix will be added to the dictionary prefix within the
                context.

        Yields:
            ParameterDict: Self with temporary prefix applied.

        """
        defaults0 = self.defaults.members
        previous0 = self.previous.members
        added_aliases = {}
        is_component = (not self.prefix)
        if is_component and cls.name:
            assert self.component is None
            self.component = cls.name
        with self.temporary_prefix(cls.name, append=True,
                                   per_member=True):
            try:
                for k, v in cls._aliases.items():
                    if k in self.provided:
                        self[v] = self.provided[k]
                        added_aliases[v] = k
                if cls._defaults and not cls.no_class_defaults:
                    self.defaults.append(cls._defaults, add_prefix=True,
                                         logger=self.logger)
                if getattr(cls, 'parameters', None):
                    self.previous.append(cls.parameters.local,
                                         add_prefix=True,
                                         logger=self.logger)
                self.add_property_dependency_defaults(cls)
                cls.add_class_defaults(self)
                yield self
            finally:
                self.defaults.members = defaults0
                self.previous.members = previous0
                if cls.initialized:
                    with self.provided.report_change(
                            prefix='Removing local: '):
                        try:
                            self.provided._immutable = False
                            for k in cls.parameters.keys():
                                if k in self.provided:
                                    self.provided.pop(k)
                                if k in added_aliases:
                                    self.provided.pop(added_aliases[k])
                        finally:
                            self.provided._immutable = True
                if is_component:
                    self.component = None

    def add_property_dependency_defaults(self, cls):
        r"""Add defaults based on the other properties that are present
        and the _property_dependencies_defaults attribute of class.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated.

        """
        if not cls._property_dependencies_defaults:
            return
        with self.report_change(prefix='Property dependency defaults: '):
            for k, v in cls._property_dependencies_defaults.items():
                if self.get(k, NoDefault) is not NoDefault:
                    continue
                for dep, cond in v['conditions'].items():
                    if cls._eval_cond(self, dep, cond):
                        self[k] = v['value']

    def copy_external_properties(self, cls):
        r"""Copy external properties identified by cls.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated.

        """
        if not cls._external_properties:
            return
        missing = []
        for k in cls._external_properties:
            self.copy_src2dst(k)
            if k not in self:
                missing.append(k)
        if not missing:
            return
        with self.temporary_source_prefix(
                '', report_change='External properties: '):
            for k in missing:
                self.copy_src2dst(k)

    def copy_component_properties(self, cls):
        r"""Copy component properties identified by cls.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated.

        """
        component = self.component
        if not (isinstance(component, str) and cls._component_properties):
            return
        missing = []
        for k in cls._component_properties:
            self.copy_src2dst(k)
            if k not in self:
                missing.append(k)
        if not missing:
            return
        with self.temporary_source_prefix(
                '', report_change='Component properties: '):
            for k in missing:
                self.copy_src2dst(k, f'{component}{k}')

    @property
    def local_parameters(self):
        r"""dict: Parameters with the current prefix."""
        # TODO: Remove keys based on property dependencies?
        out = self.select_prefix(self.prefix, strip=True)
        out = PrefixedDict(out)
        return out


class ParameterCache(SimpleWrapper):
    r"""Container for a cache of calculated parameters.

    Attributes:
         param (str): Name of parameter being generated.
         index (ParameterIndex): Index parameters are being generated for.

    """

    def __init__(self):
        self.param = None
        self.index = ParameterIndex()
        self._index_stack = []
        self._param_stack = []
        super(ParameterCache, self).__init__()

    def push_index(self, idx):
        r"""Update the index, pushing the current index onto the stack.

        Args:
            idx (ParameterIndex): New index.

        """
        self._index_stack.append(self.index.__copy__())
        self.index.update(**idx)

    def pop_index(self):
        r"""Revert the index to the last index in the stack, discarding
        the current index.

        Returns:
            ParameterIndex: Index replaced by the last index in the
                stack.

        """
        out = self.index.__copy__()
        self.index.update(**self._index_stack.pop())
        return out

    def push_param(self, param):
        r"""Update the parameter name, pushing the current parameter name
        onto the stack.

        Args:
            param (str): New parameter.

        """
        self._param_stack.append(self.param)
        self.param = param

    def pop_param(self):
        r"""Revert the parameter name to the last name in the stack,
        discarding the current name.

        Returns:
            str: Parameter name replaced by the last name in the stack.

        """
        out = self.param
        self.param = self._param_stack.pop()
        return out

    @contextlib.contextmanager
    def temporary_index(self, idx):
        r"""Context with an updated index.

        Args:
            idx (ParameterIndex): New index to use within the context.

        """
        self.push_index(idx)
        try:
            yield
        finally:
            self.pop_index()

    @contextlib.contextmanager
    def temporary_param(self, param, idx=None):
        r"""Context with an updated parameter name.

        Args:
            param (str): New parameter name to use within the context.
            idx (ParameterIndex, optioinal): New index to use within the
                context.

        """
        self.push_param(param)
        if idx is not None:
            self.push_index(idx)
        try:
            yield
        finally:
            self.pop_param()
            if idx is not None:
                self.pop_index()

    def _forward_key(self, k):
        return (k, self.index)

    def _reverse_key(self, k):
        return k[0]


class ParameterIndex(SimpleWrapper):
    r"""Container for index variables.

    Args:
        time (float, units.Quantity): Time since planting.
        age (float, units.Quantity): Age.
        n (int): Phytomer number.
        b (int): Branch level.
        x (float): Distance along current component.
        w (int): Index within component whorl.
        unit_system (, optional): Unit system that should be used.

    ClassAttributes:
        parameter_names (list): Variables that are used by the index.

    """

    universal_param = ['Time', 'Age', 'N', 'B']
    parameter_names = ['Time', 'Age', 'N', 'B', 'X', 'W']
    parameter_names_units = {
        'Time': 'time',
        'Age': 'time',
    }

    def __init__(self, *args, **kwargs):
        unit_system = kwargs.pop('unit_system', None)
        wrapped = {}
        for k, v in zip(self.parameter_names, args):
            wrapped[k] = v
        for k in self.parameter_names[len(args):]:
            wrapped[k] = kwargs.pop(k.lower(), None)
        for k in self.parameter_names:
            if isinstance(wrapped[k], tuple):
                wrapped[k] = rapidjson.units.Quantity(*wrapped[k])
        for k, kunit in self.parameter_names_units.items():
            if ((wrapped[k] is not None and unit_system is not None
                 and not isinstance(wrapped[k], rapidjson.units.Quantity))):
                wrapped[k] = rapidjson.units.Quantity(
                    wrapped[k], getattr(unit_system, kunit))
        if kwargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs}')
        wrapped = {k.lower(): v for k, v in wrapped.items()}
        super(ParameterIndex, self).__init__(wrapped)

    def __repr__(self):
        out = []
        for k in self.parameter_names:
            v = self._wrapped[k.lower()]
            if k not in self.universal_param:
                if v is not None:
                    out.append(f'{k.lower()}={v}')
            else:
                out.append(str(v))
        return ','.join(out)

    def __getitem__(self, k):
        if k.title() not in self.parameter_names:
            raise KeyError(k)
        return super(ParameterIndex, self).__getitem__(k.lower())

    def __setitem__(self, k, v):
        if k.title() not in self.parameter_names:
            raise KeyError(k)
        super(ParameterIndex, self).__setitem__(k.lower(), v)

    def __copy__(self):
        return ParameterIndex(**{k.lower(): v for k, v in self.items()})

    @property
    def tuple(self):
        r"""tuple: Tuple representation of the index."""
        out = {k.title(): self[k] for k in self.keys()}
        for k in self.parameter_names_units.keys():
            if isinstance(out[k], rapidjson.units.Quantity):
                out[k] = (float(out[k]), str(out[k].units))
        return tuple([out[k] for k in self.parameter_names])

    def __hash__(self):
        return hash(self.tuple)


class ParameterKey:
    r"""Container for parameter key for caching the current search.

    Args:
        param (str): Parameter being generated.
        idx (ParameterIndex): Parameter index.

    """

    def __init__(self, param=None, idx=None):
        self.param = param
        if idx is None:
            idx = ParameterIndex()
        self.index = idx

    def __hash__(self):
        return hash((self.param, self.index))


class SimplePlantParameter(PlantParameterBase):
    r"""Class for simple parameters requiring a single value.

    Args:
        name (str, optional): Parameter name without prefix.
        param (dict, optional): Parameters (specified via full parameter
            names with prefixes) that this parameter should take
            properties from. If not provided and parent is not provided,
            kwargs will be used.
        parent (PlantParameterBase, optional): Parent parameter that
            uses this parameter.
        defaults (dict, optional): Defaults (with only this parameter's
            prefixes) that should be used for parameters missing from
            param.
        seed (int, optional): Seed that should be used for the random
            number generator to make geometries reproducible.
        verbose (bool, optional): If True, log messages will be printed
            describing the generation process.
        debug (bool, optional): If True, pdb break points will be set
            for debug messages and log messages will be displayed.
        debug_param (list, optional): Parameters (expressed as full names)
            that debug mode should be turned on for (only exact parameter
            matches are debugged, not children).
        debug_param_prefix (list, optional): Prefix of parameters that
            debug mode should be turned on for.
        required (bool, optional): If True, this parameter is required
            and errors will be raised if it is not fully specified.
        **kwargs: Additional keyword arguments are only allowed for
            root parameters (that don't have a parent) in which case
            they are treated as param.

    Attributes:
        parameters (dict): Child parameters used by this parameter.

    ClassAttributes:
        _name (str): Name that should be used to register the class and
            be used as the default prefix.
        _properties (dict): Set of properties that control the parameter.
        _defaults (dict): Set of defaults for properties controling the
            parameter.
        _aliases (dict): Set of aliases mapping between alias names for
            parameters and the parameters they map to.
        _required (list): Set of properties that are required by this
            parameter.
        _variables (list): Properties that reference other variables.
        _constants (dict): Properties that should be added to parsed
            parameters.
        _dependencies (list): Root level properties that this parameter
            uses.
        _subschema_keys (list): JSON schema properties that contain
            schemas.
        _component_properties (list): Properties that can be defined for
            all parameters associated with a component.
        _components (list): Plant components that may be used with
            _component_properties.

    """

    _name = 'simple'
    _help = None
    _properties = {
        '': {
            'type': ['string', 'null', 'number', 'boolean']
        },
    }
    _aliases = {}
    _required = []
    _variables = []
    _constants = {}
    _specialized = []
    _dependencies = []
    _subschema_keys = ['oneOf', 'allOf']
    _external_properties = []
    _component_properties = []
    _index_properties = []
    _components = {}

    def __init__(self, name, param, parent, required=False):
        super(SimplePlantParameter, self).__init__()
        assert isinstance(param, ParameterDict)
        self._generators = {}
        self.initialized_lpy_context = False
        self.name = name
        self.parent = parent
        self.required = required
        self.initialized = False
        self.child_parameters = self.parameter_names(
            self.fullname, 'children')
        self.core_paremeters = self.parameter_names(
            self.fullname, 'core')
        self.valid_parameters = (
            self.child_parameters + self.core_paremeters
        )
        self.parameters = ParameterValues(self)
        self.update(param)

    def clear(self):
        r"""Clear the parameter class."""
        self.initialized = False
        self.parameters.clear()

    def update(self, param):
        r"""Update the parameters.

        Args:
            param (dict): New parameters.

        """
        # in_init = (len(self.parameters.local) == 0)
        if not isinstance(param, ParameterDict):
            param = ParameterDict(param)
        param0 = param.provided.flattened
        keys0 = list(param0.keys())
        schema = self.schema()
        self.initialization_error = None
        self.log(f'PROVIDED:\n{pprint.pformat(param)}')
        with param.updating(self):
            self.parameters.clear()
            self.initialized = self._parse_single(param, schema)
        if self.initialized:
            self.log(f'Parsed param:\n{pprint.pformat(self.parameters)}')
        elif self.required:
            self.error(rapidjson.NormalizationError,
                       f'Failed to initialize required parameter:\n'
                       f'{self.initialization_error}')
        else:
            self.log('MISSING PARAMETERS')
        if self.initialized:
            self.log(f'BEFORE:\n{pprint.pformat(param0)}')
            self.log(f'SCHEMA:\n{pprint.pformat(schema)}')
            self.log(f'AFTER:\n{pprint.pformat(param.provided)}')
        else:
            DictWrapper.assert_keys_match(param.provided, keys0,
                                          logger=self)
        if self.parent is None and param.provided:
            self.debugging = True
            self.error(AssertionError,
                       f'Unparsed param:\n{pprint.pformat(param.provided)}')
        if self.initialized:
            msg = f'INITIALIZED:\n{pprint.pformat(self.contents)}'
            # debug = ((not in_init) or self.parent == self.root)
            self.log(msg, debug=self.is_debug_param)

    @classmethod
    def _get_default(cls, name, param=None):
        if name in cls._defaults:
            return cls._defaults[name]
        property_entry = cls._properties[name]
        if isinstance(property_entry, type):
            if '' in property_entry._properties:
                return property_entry._get_default('')
            elif param is not None:
                with temporary_prefix(param, name, append=True):
                    if all(property_entry.is_enabled(kreq, param)
                           for kreq in property_entry._required):
                        return True
        elif 'default' in property_entry:
            return property_entry['default']
        return NoDefault

    @classmethod
    def _eval_cond(cls, param, k, cond, prefix='', value=None,
                   include_defaults=False):
        k = f'{prefix}{k}'
        default = NoDefault
        if include_defaults:
            default = cls._get_default(
                k, param=(None if isinstance(cond, list) else param),
            )
        if value is None:
            value = (
                param.get(k, default)
                if isinstance(param, (dict, DictWrapper))
                else getattr(param, k, default)
            )
        return (((cond is True and value is not NoDefault)
                 or (cond is False and value is NoDefault)
                 or (isinstance(cond, (list, tuple))
                     and value not in cond)))

    @classmethod
    def is_enabled(cls, name, param, prefix='', include_defaults=False):
        r"""Check if a parameter is enabled based on other provided
        parameters.

        Args:
            name (str): Parameter name.
            param (dict, object): Object to retrieve parameter values
                from.
            prefix (str, optional): Prefix for parameter names.
            include_defaults (bool, optional): Consider defaults when
                 determining if a parameter is enabled.

        Returns:
            bool: True if the parameter is enabled, False otherwise.

        """
        assert name in cls._properties
        if name not in cls._property_dependencies:
            return True
        for k, v in cls._property_dependencies[name].items():
            if not cls._eval_cond(param, k, v, prefix=prefix,
                                  include_defaults=include_defaults):
                return False
        return True

    @classmethod
    def add_class_defaults(cls, param, **kwargs):
        r"""Update defaults based on parameters.

        Args:
            param (ParameterDict): Parameter set to update.
            **kwargs: Additional keyword arguments are added to param.

        """
        assert isinstance(param, ParameterDict)
        with param.report_change(prefix='kwargs: '):
            param.update(kwargs)

    @classmethod
    def schema_contains(cls, schema, key, value=None):
        r"""Check if a schema contains a property corresponding to the
        provided key.

        Args:
            schema (dict): Schema to check.
            key (str): Property to check for.
            value (object, optional): Value associated with the key.

        Returns:
            bool: True if schema contains a property for key.

        """
        if key in schema.get('properties', {}):
            return True
        if key in schema.get('required', []):
            return True
        for ksub in cls._subschema_keys:
            for v in schema.get(ksub, []):
                if cls.schema_contains(v, key, value=value):
                    return True
        return False

    @classmethod
    def specialize(cls, name, exclude=[], **kwargs):
        r"""Created a specialized version of this class.

        Args:
            name (str): Name of the root property for the class.
            exclude (list, optional): Properties that should be excluded.
            properties (dict, optional): Properties that should be added.
            defaults (dict, optional): Defaults that should be added.
            property_dependencies (dict, optional): Property dependencies
                that should be added.
            **kwargs: Additional keyword arguments are added as attributes
                to the resulting class.

        """
        exclude = exclude + cls._specialized
        if 'constants' in kwargs:
            exclude += list(kwargs['constants'].keys())
        class_dict = copy.deepcopy(kwargs)
        class_dict.setdefault('_name', name)
        class_dict.setdefault('_registry_name', f'{name}{uuid.uuid4()}')
        update_attr = [
            'properties', 'required', 'variables', 'constants',
            'dependencies', 'defaults',
            'property_dependencies',
            'property_dependencies_defaults',
            'component_properties',
            'external_properties',
            'index_properties',
        ]
        for k in update_attr:
            kattr = f'_{k}'
            vattr = getattr(cls, kattr)
            if isinstance(vattr, list):
                class_dict.setdefault(
                    kattr, [
                        k for k in vattr if k not in exclude
                    ]
                )
            elif isinstance(vattr, dict):
                class_dict.setdefault(
                    kattr, {
                        k: copy.deepcopy(v) for k, v in vattr.items()
                        if k not in exclude
                    }
                )
            if k in ['property_dependencies_defaults'] and exclude:
                for v in class_dict[kattr].values():
                    v['conditions'] = {
                        kk: vv for kk, vv in v['conditions'].items()
                        if kk not in exclude
                    }
            v = class_dict.pop(k, None)
            if not v:
                continue
            if isinstance(v, list):
                assert isinstance(class_dict[kattr], list)
                for x in v:
                    if x not in class_dict[kattr]:
                        class_dict[kattr].append(x)
            elif isinstance(v, dict):
                assert isinstance(class_dict[kattr], dict)
                class_dict[kattr].update(**v)
            else:
                raise TypeError(type(v))
        return RegisteredMetaClass(f'Created{name}', (cls,), class_dict)

    @property
    def prefix(self):
        r"""str: Prefix that child parameters will have."""
        return self.name

    def remove_prefix(self, k, use_fullname=False):
        r"""Remove the current parameter prefix from a string if it
        matches.

        Args:
            k (str): String to remove prefix from.
            use_fullname (bool, optional): If True, the full name should
                be used as the prefix.

        Returns:
            str: String with the prefix removed.

        Raises:
            ValueError: If k does not start with the prefix.

        """
        if use_fullname:
            prefix = self.fullname
        else:
            prefix = self.prefix
        if not prefix:
            return k
        if k.startswith(prefix):
            return k.split(prefix, 1)[-1]
        raise ValueError(f"String \"{k}\" does not start with the "
                         f"prefix \"{prefix}\"")

    def prefixes(self, k, use_fullname=False):
        r"""Check if a string starts with the prefix for this parameter.

        Args:
            k (str): String to check.
            use_fullname (bool, optional): If True, the full name should
                be used as the prefix.

        Returns:
            bool: True if k starts with this parameter's prefix.

        """
        if use_fullname:
            prefix = self.fullname
        else:
            prefix = self.prefix
        if not prefix:
            return True
        return k.startswith(prefix)

    @property
    def fullname(self):
        r"""str: Full name with parent prefix."""
        out = self.prefix
        if self.parent is not None:
            out = f'{self.parent.fullname}{out}'
        return out

    @property
    def component(self):
        r"""ComponentBase: Component that this property belongs to."""
        if self.parent:
            return self.parent.component
        return None

    @property
    def component_name(self):
        r"""str: Name of the component that this property belongs to."""
        component = self.component
        if component is not None:
            return component.name
        return None

    @property
    def root(self):
        r"""PlantParameterBase: Root parameter."""
        if self.parent is not None:
            return self.parent.root
        return self

    @property
    def cache(self):
        r"""ParameterCache: Cache of computed parameters."""
        if self.parent is not None:
            return self.root._cache
        return self._cache

    @contextlib.contextmanager
    def temporary_index(self, *args, **kwargs):
        r"""Context with an updated index.

        Args:
            *args, **kwargs: Arguments are used to create a
                ParameterIndex instance for the context.

        """
        idx = ParameterIndex(*args, **kwargs)
        for k, v in self.current_index.items():
            if idx[k] is None:
                idx[k] = v
        with self.cache.temporary_index(idx):
            yield

    @property
    def current_param(self):
        r"""str: Current parameter being generated."""
        return self.cache.param

    @property
    def current_index(self):
        r"""ParameterIndex: Current parameter index."""
        return self.cache.index

    @property
    def local_index(self):
        r"""ParameterIndex: Local parameter index with unused parameters
        nullified."""
        out = copy.copy(self.current_index)
        for k in self.constants_index:
            out[k] = None
        return out

    @property
    def verbose(self):
        r"""bool: If True, log messages will be emitted."""
        if self.debugging:
            return True
        if self.parent is not None:
            return self.root._verbose
        return self._verbose

    @property
    def no_class_defaults(self):
        if self.parent is not None:
            return self.root._no_class_defaults
        return self._no_class_defaults

    @property
    def debug_param(self):
        r"""list: Parameters that debug mode should be enabled for."""
        if self.parent is not None:
            return self.root._debug_param
        return self._debug_param

    @property
    def debug_param_prefix(self):
        r"""list: Parameters that debug mode should be enabled for."""
        if self.parent is not None:
            return self.root._debug_param_prefix
        return self._debug_param_prefix

    @property
    def is_debug_param(self):
        r"""bool: True if this is a parameter being debugged."""
        if self.fullname.startswith(tuple(self.debug_param_prefix)):
            return True
        if self.fullname in self.debug_param:
            return True
        return False

    @property
    def debugging(self):
        r"""bool: True if debugging is active."""
        if self.is_debug_param:
            return True
        if self.parent is not None:
            return self.root._debugging
        return self._debugging

    @debugging.setter
    def debugging(self, value):
        if self.parent:
            self.root.debugging = value
            return
        self._debugging = value
        if value:
            self.log('Turning on debugging')

    def debug(self, message='', force=False, **kwargs):
        r"""Set a pdb break point if debugging is active.

        Args:
            message (str, optional): Log message to show before setting
                break point.
            force (bool, optional): If True, the break point will be
                set even if debugging is not active.
            **kwargs: Additional keyword arguments are passed to the
                log method if it is called.

        """
        self.log(f'DEBUG: {message}', force=force, **kwargs)
        if not (force or self.debugging):
            return
        pdb.set_trace()

    def lsystem_args(self, for_rule=False, **kwargs):
        r"""Get the arguments that should be used for Lsystem rules.

        Args:
            for_rule (bool, optional): If True, all parameters are
                positional.
            **kwargs: Additional keyword arguments are checked for
                increments that should be applied to arguments with keys
                of the form '{name}inc'.

        Returns:
            str: Arguments.

        """
        try:
            args = [k.lower() for k in ParameterIndex.universal_param]
            args_opt = [k.lower() for k in self.dependencies_index
                        if k.lower() not in args + ['x']]
            values = {k: k for k in args + args_opt}
            for kinc, v in kwargs.items():
                assert kinc.endswith('inc')
                k = kinc.rsplit('inc', maxsplit=1)[0]
                if v is True:
                    values[k] += ('+ageinc' if k == 'age' else '+1')
                elif isinstance(v, (str, int)):
                    values[k] += f'+{v}'
            out = [values[k] for k in args]
            if for_rule:
                out += [values[k] for k in args_opt]
            else:
                out += [f'{k}={values[k]}' for k in args_opt]
            return ','.join(out)
        except AttributeError:
            import traceback
            print("LSYSTEM_ARGS")
            print(traceback.format_exc())
            raise

    # Methods for generating parameters
    def hasmethod(self, k):
        r"""Check if there is an explicit method for a parameter.

        Args:
            k (str): Parameter name.

        Returns:
            bool: True if there is an explicit method, False otherwise.

        """
        return hasattr(type(self), k)

    def __getattr__(self, k):
        if self.hasmethod(k):
            raise AttributeError(f'{self}: {k}')
        if self.parent is not None:
            self.error(AttributeError, f'{self}: {k}')
        if k not in self.valid_parameters:
            self.error(AttributeError, f'{self}: {k}')
        return functools.partial(self.getfull, k, strip_units=True)

    def getfull(self, k, *args, **kwargs):
        r"""Get a parameter value with the index fully specified as
        arguments.

        Args:
            k (str): Name of parameter to return without any prefixes.
            age (float): Age of the component that the parameter should
                be returned for.
            n (int): Phytomer count of the component that the parameter
                should be returned for.
            x (float, optional): Position along the component that the
                parameter should be returned for.
            w (int, optional): Whorl index of repeated component that the
                parameter should be returned for.
            strip_units (bool, optional): If True, strip units from
                quantities after converting to the desired unit system.
            **kwargs: Additional keyword arguments are passed to get.

        Returns:
            object: Parameter value.

        """
        strip_units = kwargs.pop('strip_units', False)
        kws_index = {
            k.lower(): kwargs.pop(k.lower()) for k in
            ParameterIndex.parameter_names
            if k.lower() in kwargs
        }
        kws_index['unit_system'] = self.unit_system
        with self.temporary_index(*args, **kws_index):
            out = self.get(k, **kwargs)
        if strip_units:
            out = self.unit_system.convert(out, strip=True)
        return out

    def set(self, k, value):
        r"""Set a parameter value.

        Args:
            k (str): Name of parameter to set, without any prefixes.
            value (object): Value to set parameter to.

        """
        raise NotImplementedError

    @property
    def children(self):
        r"""iterable: Child parameter instances."""
        for v in self.parameters.nested.values():
            yield v

    @cached_property
    def component_parameters(self):
        r"""list: Set of variables that this parameter uses from the
        parent component."""
        return copy.deepcopy(self._component_properties)

    @cached_property
    def variable_parameters(self):
        r"""list: Set of external parameters identified by variables."""
        out = []
        for k in self._variables:
            if k not in self.parameters:
                continue
            kv = self.parameters[k]
            if isinstance(kv, list):
                out += kv
            else:
                out.append(kv)
        return out

    @cached_property
    def index_parameters(self):
        r"""list: Set of index parameters that this parameter uses."""
        out = copy.deepcopy(self._index_properties)
        out += [k for k in self.variable_parameters
                if k in ParameterIndex.parameter_names]
        for k in self.external_parameters:
            v = self.get(k, return_other='instance')
            if isinstance(v, PlantParameterBase):
                out += v.index_parameters
        for child in self.children:
            out += child.index_parameters
        return list(set(out))

    @cached_property
    def external_parameters(self):
        r"""list: Set of variables that this parameter uses from the
        root generator."""
        out = copy.deepcopy(self._external_properties)
        out += [k for k in self.variable_parameters
                if k not in ParameterIndex.parameter_names]
        out += [
            f'{self.component_name}{k}'
            for k in self.component_parameters
        ]
        return list(set(out))

    @cached_property
    def dependencies(self):
        r"""list: Set of variables that this parameter is dependent on."""
        out = copy.deepcopy(self.external_parameters)
        for k in self.external_parameters:
            v = self.get(k, return_other='instance')
            if isinstance(v, PlantParameterBase):
                out += v.dependencies
        for child in self.children:
            out += child.dependencies
        return list(set(out))

    @cached_property
    def all_parameters(self):
        r"""dict: Set of all parameters, including children."""
        out = PrefixedDict({}, prefix=self.prefix)
        for k, v in self.parameters.items():
            if isinstance(v, PlantParameterBase):
                assert v.initialized
                out.update(**v.all_parameters)
            else:
                out[k] = v
        return out.raw_flattened

    @cached_property
    def constants_index(self):
        r"""list: Set of index variables that this parameter is
        independent of."""
        return [k for k in ParameterIndex.parameter_names
                if k not in self.index_parameters]

    @cached_property
    def dependencies_index(self):
        r"""list: Set of index variables that this parameter is
        dependent on."""
        return [k for k in ParameterIndex.parameter_names
                if k in self.index_parameters]

    def _nullify_constant_var(self, idx, var=None, context=None,
                              make_copy=False):
        if make_copy:
            idx = copy.copy(idx)
        if var is None:
            var = self.constants_index
        if var is None:
            return idx
        if isinstance(var, list):
            for v in var:
                idx = self._nullify_constant_var(idx, var=v,
                                                 context=context)
            return idx
        idx[var] = None
        return idx

    @classmethod
    def get_class(cls, k, prefix=''):
        r"""Get a parameter class.

        Args:
            k (str): Name of parameter to return, without any prefixes.

        Returns:
            PlantParameterBase: Parameter class.

        """
        if prefix and not k.startswith(prefix):
            raise KeyError(f'\'{k}\' does not start with prefix '
                           f'\'{prefix}\'')
        if prefix:
            k = k.split(prefix, 1)[-1]
        if k in cls._properties:
            return cls._properties[k]
        for kk, vv in cls._properties.items():
            if isinstance(vv, dict):
                continue
            try:
                return vv.get_class(k, prefix=kk)
            except KeyError:
                pass
        raise KeyError(k)

    def get(self, k, default=NoDefault, **kwargs):
        r"""Get a parameter value.

        Args:
            k (str): Name of parameter to return, without any prefixes.
            default (object, optional): Default to return if the
                parameter is not set.
            **kwargs: Additional keyword arguments are passed to
                parameters.get.

        Returns:
            object: Parameter value.

        """
        return self.parameters.get(k, default=default, **kwargs)

    def getold(self, k, default=NoDefault, idx=None, cache_key=None,
               k0=None, return_other=None, **kwargs):
        r"""Get a parameter value.

        Args:
            k (str): Name of parameter to return, without any prefixes.
            default (object, optional): Default to return if the
                parameter is not set.
            idx (ParameterIndex, optional): Simulation index to get the
                parameter value for. If not provided, the current index
                will be used based on the last parameter accessed with an
                index.
            cache_key (tuple, optional): Key to use to cache any
                generated values if different than the default.
            k0 (str, optional): Full name of the parameter including any
                parent prefixes. If not provided, the full name will be
                created based on the current prefix.
            return_other (str, optional): Name of what should be
                returned instead of the parameter value.
            **kwargs: Additional keyword arguments are passed to any
                nested calls to get or generate.

        Returns:
            object: Parameter value.

        """
        if idx is None:
            idx = self._current_idx(cache_key=cache_key)
        if return_other:
            kidx = idx
        else:
            self.log(f'dependencies = {self.dependencies}')
            kidx = self._nullify_constant_var(idx, context=k, make_copy=True)
        if k0 is None:
            is_child = True
            k0 = f'{self.fullname}{k}'
        else:
            is_child = self.prefixes(k)
            if is_child:
                k = self.remove_prefix(k)
        if k in kidx.parameter_names:
            try:
                return kidx[k]
            except BaseException as e:
                self.error(KeyError, str(e))
        elif k in self._constants:
            return self._constants[k]
        elif k0 not in self.valid_parameters:
            if is_child:
                if default is not NoDefault:
                    return default
                self.error(KeyError,
                           f'Missing child parameter \"{k}\" '
                           f'(k0=\"{k0}\")?', debug=True)
            self.log(f'External parameter \"{k0}\"')
            return self.root.get(k0, default=default, idx=idx,
                                 return_other=return_other, **kwargs)
        if k and k not in self.parameters:
            for kk, vv in self.parameters.items():
                if isinstance(vv, PlantParameterBase) and vv.prefixes(k):
                    return vv.get(k, idx=kidx, k0=k0,
                                  cache_key=cache_key, default=default,
                                  return_other=return_other, **kwargs)
            else:
                if default is not NoDefault:
                    return default
                self.error(KeyError, f"Unsupported var \"{k0}\"")
        if not return_other:
            out = self._push_key(k0, kidx, cache_key=cache_key)
            if out is not None:
                return out
        try:
            if k == '':
                v = self
            elif k in self.parameters:
                v = self.parameters[k]
            else:
                self.error(KeyError, f"Unsupported var? \"{k0}\" (\"{k}\")")
            if isinstance(v, PlantParameterBase):
                if not v.initialized:
                    self.error(KeyError, f"Parameter not initialized \"{k0}\"")
                if return_other is None:
                    self.log(f'Generating \"{k0}\"')
                    v = v.generate(kidx, **kwargs)
                elif return_other == 'instance':
                    pass
                elif hasattr(v, return_other):
                    v = getattr(v, return_other)
                else:
                    raise ValueError(f'Invalid return_other = '
                                     f'\"{return_other}\"')
        except BaseException:
            if default is NoDefault:
                raise
            v = default
        if not return_other:
            self._pop_key(v)
        return v

    def generate(self, **kwargs):
        r"""Generate this parameter, updating the cache as necessary.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        with self.cache.temporary_param(self.fullname, self.local_index):
            if kwargs or self.fullname not in self.cache:
                try:
                    self.log('GENERATING')
                    v = self._generate(**kwargs)
                    if kwargs:
                        self.log(f'GENERATED [UNCACHED]: {v}')
                        return v
                    self.cache[self.fullname] = v
                    self.log(f'GENERATED: {v}')
                except BaseException as e:
                    self.debug('ERROR IN GENERATE', force=True,
                               exception=e)
                    raise
            return self.cache[self.fullname]

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        return self.get('')

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        out = {
            'type': 'object',
            'properties': copy.deepcopy(cls._properties),
        }
        if cls._required:
            out['required'] = cls._required.copy()
        return out

    @classmethod
    def parameter_names(cls, name, scope='all'):
        r"""Get a list of valid parameter names.

        Args:
            name (str): Base name for parameters.
            scope (str, optional): Scope for parameters that should
                be returned. Options are:
                  'all': Return all parameters.
                  'core': Return parameters only directly allowed by
                      this parameter.
                  'children': Return parameters only indirectly allowed
                      by child parameters.

        Returns:
            list: Parameter names.

        """
        out = []
        if scope in ['all', 'core']:
            out += [f'{name}{k}' for k in cls._properties.keys()]
            out += [f'{name}{k}' for k in cls._constants.keys()]
            if name not in out:
                out.insert(0, name)
        if scope in ['all', 'children']:
            for k, v in cls._properties.items():
                if isinstance(v, type):
                    kname = f'{name}{k}'
                    out += v.parameter_names(kname)
        return out

    def _extract_parameters(self, schema, param):
        parameters = {'properties': {}, 'required': []}
        for k in list(schema.get('properties', {}).keys()):
            if isinstance(schema['properties'][k], type):
                parameters['properties'][k] = schema['properties'].pop(k)
                if k in schema.get('required', []):
                    parameters['required'].append(k)
                    schema['required'].remove(k)
            elif k:
                keys = [k for k in param.keys(prefix=k)
                        if (k not in parameters['properties']
                            and k not in schema['properties'])]
                if len(keys) > 1:
                    parameters['properties'][k] = schema2parameter(
                        k, schema['properties'].pop(k))
                    if k in schema.get('required', []):
                        parameters['required'].append(k)
                        schema['required'].remove(k)
        if not schema.get('properties', {}):
            schema.pop('properties', None)
        if not schema.get('required', []):
            schema.pop('required', None)
        if not parameters['properties']:
            parameters.pop('properties')
        if not parameters['required']:
            parameters.pop('required')
        return parameters

    def _parse_single(self, param, schema, child_param=None,
                      parameters=None, required=None, idstr='',
                      subschema_errors=None):
        if idstr:
            self.log(f'{idstr}: {schema}')
        if child_param is None:
            child_param = param.local_parameters
            self.log(f'CHILDREN:\n{pprint.pformat(child_param)}')
        if required is None:
            required = self.required
        # Parse subschemas w/ parameter classes via recursion
        try:
            for k in self._subschema_keys:
                if k not in schema:
                    continue
                krequired = (k == 'allOf')
                if k == 'oneOf':
                    keys_all = set()
                    for x in schema[k]:
                        keys_all
                kparam = [
                    self._extract_parameters(x, child_param)
                    for x in schema[k]
                ]
                if not any(kparam):
                    continue
                errors = []
                results = [
                    self._parse_single(
                        param, x, child_param=child_param,
                        idstr=f' {k}[{i}]', required=krequired,
                        parameters=kparam[i], subschema_errors=errors,
                    )
                    for i, x in enumerate(schema[k])
                ]
                if k == 'oneOf':
                    if sum(results) > 1:
                        valid = [
                            schema[k][i] for i, v in
                            enumerate(results) if v
                        ]
                        raise rapidjson.NormalizationError(
                            f'More than one match ({len(valid)}):\n'
                            f'{pprint.pformat(valid)}\n'
                            f'{pprint.pformat(child_param)}'
                        )
                    elif sum(results) == 0:
                        raise rapidjson.NormalizationError(
                            f'No matches\n{"\n".join(errors)}')
                elif k == 'allOf':
                    if sum(results) != len(results):
                        invalid = [
                            schema[k][i] for i, v in
                            enumerate(results) if not v
                        ]
                        raise rapidjson.NormalizationError(
                            f'Not all match:\n'
                            f'{pprint.pformat(invalid)}\n'
                            f'{pprint.pformat(child_param)}:\n'
                            f'{"\n".join(errors)}'
                        )
                else:
                    self.error(NotImplementedError,
                               f'Unsupported schema key {k}')
                schema.pop(k)
            # Split schema & param into regular parameters and parameter
            #   classes
            if parameters is None:
                parameters = self._extract_parameters(schema, child_param)
            child_param_schema = {
                k: v for k, v in child_param.items()
                if self.schema_contains(schema, k, value=v)
            }
            # Parse regular schema parameters
            if schema:
                try:
                    out = rapidjson.normalize(child_param_schema, schema)
                    self.log(f"NORMALIZE: {child_param_schema}\n{out}")
                except rapidjson.NormalizationError as e:
                    raise rapidjson.NormalizationError(
                        f'Normalization error: {e.args[0]}\n'
                        f'Schema param:\n'
                        f'{pprint.pformat(child_param_schema)}'
                    )
            # Parse parameter classes
            missing = []
            missing_errors = []
            for k, v in parameters.get('properties', {}).items():
                self.log(f'Adding{idstr} child parameter \"{k}\": {v}')
                krequired = (k in parameters.get('required', []))
                if isinstance(child_param.get(k, None),
                              PlantParameterBase):
                    out[k] = child_param.pop(k)
                    out[k].update(param)
                else:
                    out[k] = v(
                        k, param, self, required=(required and krequired),
                    )
                if not out[k].initialized:
                    if krequired:
                        missing.append(k)
                        missing_errors.append(out[k].initialization_error)
                    out.pop(k)
            if missing:
                raise rapidjson.NormalizationError(
                    f'Failed to initialized child parameter '
                    f'instance(s): {missing}\n'
                    f'{"\n".join(missing_errors)}'
                )
            # Update parameters on the class
            self.log(f"ADDING:\n{pprint.pformat(out)}")
            self.parameters.update(**out)
            return True
        except rapidjson.NormalizationError as e:
            if required:
                self.error(rapidjson.NormalizationError, e.args[0])
            self.log(f'Ignored error: {e.args[0]}')
            if subschema_errors is not None:
                subschema_errors.append(e.args[0])
            else:
                self.initialization_error = e.args[0]
            return False

    # Logging methods
    @property
    def contents(self):
        r"""dict: The contents of parameters."""
        param = {}
        for k, v in self.parameters.items():
            if isinstance(v, PlantParameterBase):
                if v.initialized:
                    param[k] = v.contents
            else:
                param[k] = v
        return param

    @property
    def log_prefix_instance(self):
        r"""str: Prefix to use for log messages emitted by this instance."""
        out = ''
        if self.parent:
            out = f'{self.parent.log_prefix_instance}->'
        if self.prefix:
            out += self.prefix
        else:
            out += self._name
        return out

    @property
    def log_prefix_stack(self):
        r"""str: Prefix to use for log messages describing the current
        stack."""
        if self.current_param is None:
            return ''
        return f'{self.current_param}[{self.current_index}]: '

    @property
    def log_prefix(self):
        r"""str: Prefix to add to messages emitted by this instance."""
        return f'{self.log_prefix_instance}: {self.log_prefix_stack}'

    def log(self, message='', debug=False, **kwargs):
        r"""Emit a log message.

        Args:
            message (str, optional): Log message.
            force (bool, optional): If True, print the log message even
                if self.verbose is False.
            debug (bool, optional): If True, set a debug break point if
                debugging enabled.
            **kwargs: Additional keyword arguments are passed to the
                parent method.

        """
        return super(SimplePlantParameter, self).log(
            message=message, debug=(debug and self.debugging), **kwargs
        )

    def error(self, error_cls, message='', debug=False, **kwargs):
        r"""Raise an error, adding context to the message.

        Args:
            error_cls (type): Error class.
            message (str, optional): Error message.
            debug (bool, optional): If True, set a debug break point.
            **kwargs: Additional keyword arguments are passed to the
                parent method.

        """
        return super(SimplePlantParameter, self).error(
            error_cls, message=message,
            debug=(debug or self.debugging),
            **kwargs
        )

    @property
    def generator(self):
        r"""np.random.Generator: Current random number generator."""
        if self.parent:
            return self.parent.generator
        seed = self.seed
        if self.current_index['N'] is not None:
            seed += self.current_index['N']
        if seed not in self._generators:
            self.log(f"Creating generator for seed = {seed}")
            self._generators[seed] = np.random.default_rng(
                seed=seed)
        return self._generators[seed]

    def sample_dist(self, profile, *args, **kwargs):
        r"""Sample a distribution using the current random number
        generator.

        Args:
            profile (str): Name of the profile that should be sampled.
            *args, **kwargs: Additional arguments are passed to the
                generator method for the specified profile.

        """
        if not args:
            args = None
        return DistributionPlantParameter.sample_generator_dist(
            self.generator, profile=profile, args=args, **kwargs)

    def initialize_lpy_context(self, context):
        r"""Initialize the lpy context for this instance.

        Args:
            context (lpy.LsysContext): Context to initialize.

        """
        if self.initialized_lpy_context:
            return
        for child in self.children:
            child.initialize_lpy_context(context)
        self.initialized_lpy_context = True

    def initialize_lpy_turtle(self, turtle):
        r"""Initialize a PglTurtle instance.

        Args:
            turtle (plantgl.PglTurtle): Turtle to update.

        """
        for child in self.children:
            child.initialize_lpy_turtle(turtle)


class OptionPlantParameter(SimplePlantParameter):
    r"""Class for parameter with exclusive options."""

    _name = None
    _name_option = ''
    _properties = {}
    _option_dependencies = {}
    _default = NoDefault

    @staticmethod
    def _on_registration(cls):
        opt = cls._name_option
        cls._properties = copy.deepcopy(cls._properties)
        cls._property_dependencies = copy.deepcopy(
            cls._property_dependencies)
        choices = list(sorted(cls._option_dependencies.keys()))
        if opt is not None and choices:
            if opt not in cls._properties:
                cls._properties[opt] = {}
            if opt not in cls._required:
                cls._required = copy.deepcopy(cls._required)
                cls._required.insert(0, opt)
            cls._properties[opt].setdefault('enum', choices)
            for k, v in cls._option_dependencies.items():
                for vv in v:
                    cls._property_dependencies.setdefault(vv, {})
                    cls._property_dependencies[vv].setdefault(opt, [])
                    cls._property_dependencies[vv][opt].append(k)
            for k in cls._properties.keys():
                if k == opt:
                    continue
                cls._property_dependencies.setdefault(k, {})
                cls._property_dependencies[k].setdefault(opt, True)
        if cls._default is not NoDefault:
            assert opt is not None
            cls._defaults = copy.deepcopy(cls._defaults)
            cls._defaults[opt] = copy.deepcopy(cls._default)
        SimplePlantParameter._on_registration(cls)

    @classmethod
    def _get_property_dependencies(cls, name, prefix='', **kwargs):
        out = super(
            OptionPlantParameter, cls)._get_property_dependencies(
                name, prefix=prefix, **kwargs)
        option_property = cls._name_option
        if name == option_property:
            if len(out) > 1:
                out[-1].update(out[-2])
                out[-2] = {}
            return out
        return out

    @classmethod
    def specialize(cls, name, exclude_options=None, **kwargs):
        r"""Created a specialized version of this class.

        Args:
            name (str): Name of the root property for the class.
            exclude_options (list, optional): Set of options to eliminate.
            **kwargs: Additional keyword arguments are added as attributes
                to the resulting class.

        """
        if exclude_options:
            kwargs['_option_dependencies'] = copy.deepcopy(
                cls._option_dependencies)
            kwargs.setdefault('exclude', [])
            for x in exclude_options:
                kwargs['exclude'] += list(
                    cls.option_parameters(x, unique=True)
                )
                kwargs['_option_dependencies'].pop(x)
        return super(OptionPlantParameter, cls).specialize(name, **kwargs)

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        opt = cls._name_option
        out = super(OptionPlantParameter, cls).schema()
        if opt is None:
            return out
        out['oneOf'] = []
        for k, v in cls._option_dependencies.items():
            kprop = {}
            krequired = out.get('required', []) + v
            assert opt in krequired
            for kreq in krequired:
                kprop[kreq] = copy.deepcopy(out['properties'][kreq])
            kprop[opt]['enum'] = [k]
            if cls._default is NoDefault:
                unique_param = cls.option_parameters(k, unique=True)
                if unique_param:
                    kprop[opt]['default'] = k
            out['oneOf'].append({
                'properties': kprop,
                'required': krequired,
            })
        option_properties = cls.option_parameters(None, unique=True)
        for k in option_properties:
            out['properties'].pop(k)
        out.pop('required', None)
        return out

    @classmethod
    def option_parameters(cls, option, unique=False, required=False):
        r"""Get the set of parameters enabled by an option.

        Args:
            option (str): Option value.
            unique (bool, optional): If True, return only those parameters
                that are unique to the provided option.

        Returns:
            set: Option parameters.

        """
        if required:
            all_options = (
                set(cls._required) - set([cls._name_option])
            )
        else:
            all_options = (
                set(cls._properties.keys()) - set([cls._name_option])
            )
        if option is None and not (unique | required):
            return all_options
        other_options = set()
        this_option = set()
        for k, v in cls._option_dependencies.items():
            vall = set(v)
            if k == option:
                this_option |= vall
            else:
                other_options |= vall
        if option is None and unique:
            return other_options
        elif option is None and required:
            return all_options | other_options
        elif unique:
            return this_option - other_options
        return (all_options - other_options) | this_option

    @classmethod
    def add_class_defaults(cls, param, **kwargs):
        r"""Update defaults based on parameters.

        Args:
            param (ParameterDict): Parameter set to update.
            **kwargs: Additional keyword arguments are added to param.

        """
        # Remove defaults added for unused options
        opt = cls._name_option
        vopt = None if opt is None else param.get(opt, NoDefault)
        unique_keys = {
            k: cls.option_parameters(k, unique=True)
            for k in cls._option_dependencies.keys()
        }
        if vopt is NoDefault:
            choices = {}
            empty = []
            for k, unique in unique_keys.items():
                count = sum([
                    x in param for x in unique
                ])
                if unique and count:
                    choices[k] = count
                elif len(cls._option_dependencies[k]) == 0:
                    empty.append(k)
            if choices:
                vopt = max(choices, key=lambda x: choices[x])
                kwargs.setdefault(opt, vopt)
            elif len(empty) == 1:
                kwargs.setdefault(opt, empty[0])
        super(OptionPlantParameter, cls).add_class_defaults(
            param, **kwargs)


class FunctionPlantParameter(OptionPlantParameter):
    r"""Class for a function."""

    _name = 'Func'
    _help = 'Function producing {parent}'
    _properties = {
        'VarName': {
            'type': ['string', 'array'],
            'description': ('Parameter(s) that serves as the {parent} '
                            'function\'s independent variable'),
        },
        'VarNorm': {
            'type': 'string',
            'description': ('Parameter that {parent}VarName variable '
                            'should be normalized by before applying '
                            'the {parent} function. If not provided, '
                            'but VarMin & VarMax are, the difference '
                            'between the two will be used.'),
        },
        'VarMin': {
            'type': 'string',
            'description': ('Parameter that specifies the minimum '
                            '{parent}VarName value for which the '
                            '{parent} function is applied.'),
        },
        'VarMax': {
            'type': 'string',
            'description': ('Parameter that specifies the maximum '
                            '{parent}VarName value for which the '
                            '{parent} function is applied.'),
        },
        'Slope': {
            'type': 'scalar', 'subtype': 'float',
            'description': 'Slope of the {parent} function',
        },
        'Intercept': {
            'type': 'scalar', 'subtype': 'float',
            'description': ('Value of the {parent} function when '
                            '{parent}VarName variable is 0'),
        },
        'Amplitude': {
            'type': 'scalar', 'subtype': 'float', 'default': 1.0,
            'description': 'Scale factor for {parent} function',
        },
        'Period': {
            'type': 'scalar', 'subtype': 'float', 'default': 2.0 * np.pi,
            'description': 'Period of {parent} sinusoid function',
        },
        'XOffset': {
            'type': 'scalar', 'subtype': 'float', 'default': 0.0,
            'description': ('Offset added to independent variable '
                            'before scaling for the period and applying '
                            'the sinusoid function'),
        },
        'YOffset': {
            'type': 'scalar', 'subtype': 'float', 'default': 0.0,
            'description': ('Offset added to the function result'),
        },
        'Exp': {
            'type': 'scalar', 'subtype': 'float',
            'description': 'Exponent',
        },
        'XVals': {
            'oneOf': [
                {'type': 'ndarray', 'subtype': 'float'},
                {'type': 'array', 'items': {
                    'type': 'scalar', 'subtype': 'float'
                },
                 'minItems': 2, 'maxItems': 2},
            ],
            'description': ('Independent variable values or range that '
                            'should be used for interpolation'),
        },
        'YVals': {
            'type': 'ndarray', 'subtype': 'float',
            'description': ('Dependent variable values that should be '
                            'used for interpolation'),
        },
        'Curve': DelayedPlantParameter(
            'curve',
            _help='Curve that should be sampled as a function',
        ),
        'CurvePatch': DelayedPlantParameter(
            'curve_patch',
            _help='Curve patch that should be sampled as a function',
        ),
        'Method': {
            'type': 'string',
            'description': ('Name of the method on the plant generator '
                            'class that should be called')
        },
        'Function': {'type': 'function'},
        'Expression': {
            'type': 'string',
            'description': ('Python expression that should be evaluated '
                            'with variables defined'),
        },
    }
    _option_dependencies = {
        'zero': [],
        'one': [],
        'alias': [],
        'linear': ['Slope', 'Intercept'],
        'sin': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'cos': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'tan': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'pow': ['Amplitude', 'Exp', 'XOffset', 'YOffset'],
        'logistic': ['Amplitude', 'Exp', 'XOffset', 'YOffset'],
        'interp': ['XVals', 'YVals'],
        'curve': ['Curve'],
        'curve_patch': ['CurvePatch'],
        'method': ['Method'],
        'user': ['Function'],
        'expression': ['Expression']
    }
    _required = ['VarName']
    _variables = ['VarName', 'VarNorm', 'VarMin', 'VarMax']

    @property
    def namevar(self):
        r"""str: Variable that this function parameter takes as input."""
        out = self.get('VarName')
        if isinstance(out, list) and len(out) == 1:
            return out[0]
        return out

    @property
    def normvar(self):
        r"""str: Variable that should be used to normalize the input."""
        return self.get('VarNorm', None)

    @property
    def maxvar(self):
        r"""str: Variable that contains the maximum value under which
        the function applies."""
        return self.get('VarMax', None)

    @property
    def minvar(self):
        r"""str: Variable that contains the minimum value under which
        the function applies."""
        return self.get('VarMin', None)

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        func = self.get('')
        if func == 'one':
            return 1.0
        elif func == 'zero':
            return 0.0
        elif func == 'expression':
            expression = self.get('Expression')
            variables = {
                k: self.get(k, None) for k in self.variable_parameters
            }
            return eval(expression, globals={}, locals=variables)
        v = self.get(self.namevar)
        if func == 'alias':
            return v
        v0 = v
        vmin = None
        vmax = None
        vnorm = None
        if self.maxvar is not None:
            vmax = self.get(self.maxvar)
            if v > vmax:
                v = vmax
        if self.minvar is not None:
            vmin = self.get(self.minvar)
            if v < vmin:
                v = vmin
        if self.normvar is not None:
            vnorm = self.get(self.normvar)
        elif vmin is not None and vmax is not None:
            v = v - vmin
            vnorm = vmax - vmin
        if vnorm is not None:
            v = v / vnorm
        if callable(func):
            out = func(v)
        elif func == 'linear':
            slope = self.get('Slope')
            intercept = self.get('Intercept')
            out = slope * v + intercept
        elif func in ['sin', 'cos', 'tan']:
            A = self.get('Amplitude')
            period = self.get('Period')
            xoffset = self.get('XOffset')
            yoffset = self.get('YOffset')
            ftrig = getattr(np, func)
            out = (
                (A * ftrig(2.0 * np.pi * (v + xoffset) / period))
                + yoffset)
        elif func == 'pow':
            A = self.get('Amplitude')
            exp = self.get('Exp')
            xoffset = self.get('XOffset')
            yoffset = self.get('YOffset')
            out = (A * pow(v + xoffset, exp)) + yoffset
        elif func == 'logistic':
            A = self.get('Amplitude')
            xoffset = self.get('XOffset')
            yoffset = self.get('YOffset')
            out = (A / (1.0 + np.exp(-(v + xoffset)))) + yoffset
        elif func == 'interp':
            xvals = self.get('XVals')
            yvals = self.get('YVals')
            if isinstance(xvals, (tuple, list)):
                # self.log(f"BEFORE: {xvals}", force=True)
                xvals = np.linspace(xvals[0], xvals[1], len(yvals))
            # self.log(f"INTERP: {xvals}", force=True)
            # out = np.interp(xval, yvals
            f = scipy.interpolate.interp1d(xvals, yvals,
                                           fill_value="extrapolate")
            out = f(v)
        elif func == 'curve':
            curve = self.get('Curve', **kwargs)
            out = CurvePlantParameter.sample_curve(curve, v)
        elif func == 'curve_patch':
            patch = self.get('CurvePatch', **kwargs)
            out = CurvePatchPlantParameter.sample_curve_patch(patch, v)
        elif func == 'method':
            method = self.get('Method', **kwargs)
            assert self.root.hasmethod(method)
            out = getattr(self.root, method)(v)
        elif func == 'user':
            function = self.get('Function', **kwargs)
            out = function(v)
        else:
            raise ValueError(f"Unsupported function name \"{func}\"")
        self.log(f'{self.fullname}[{self.namevar}={v0} (v={v})] = {out} '
                 f'(MIN: {self.minvar} = {vmin}, '
                 f'MAX: {self.maxvar} = {vmax}, '
                 f'NORM: {self.normvar} = {vnorm})')
        return out

    @classmethod
    def specialize_var(cls, var, normvar=None, minvar=None, maxvar=None,
                       **kwargs):
        r"""Created a specialized version of this class.

        Args:
            var (str): Name of the variable that the specialized
                function is dependent on.
            normvar (str, optional): Name of the variable that should be
                used to normalize the dependent variable.
            maxvar (str, optional): Name of the variable that should be
                used to cap the values of the dependent variable on
                which the function should be evaluated.
            **kwargs: Additional keyword arguments are passed to the
                base class's specialize method.

        """
        kwargs.setdefault('namevar', var)
        kwargs.setdefault('normvar', normvar)
        kwargs.setdefault('minvar', minvar)
        kwargs.setdefault('maxvar', maxvar)
        var = kwargs['namevar']
        descriptions = {
            'name': 'Indepdendent variable',
            'norm': f'Value used to normalize {var.lower()} for {{parent}}',
            'min': f'Minimum {var.lower()} over which {{parent}} is valid',
            'max': f'Maximum {var.lower()} over which {{parent}} is valid',
        }
        kwargs.setdefault('required', [])
        kwargs.setdefault('properties', {})
        kwargs.setdefault('exclude', [])
        kwargs.setdefault('external_properties', [])
        kwargs.setdefault('index_properties', [])
        for k in descriptions.keys():
            xVar = f'Var{k.title()}'
            kwargs['exclude'].append(xVar)
            x = kwargs[f'{k}var']
            if x is None:
                continue
            if x in ParameterIndex.parameter_names:
                if x not in kwargs['index_properties']:
                    kwargs['index_properties'].append(x)
            else:
                # if x not in kwargs['required']:
                #     kwargs['required'].append(x)
                if x not in kwargs['external_properties']:
                    kwargs['external_properties'].append(x)
                if x not in kwargs['properties']:
                    kwargs['properties'][x] = {
                        'type': 'scalar', 'subtype': 'float',
                        'description': descriptions[k],
                    }
        return FunctionPlantParameter.specialize(f'{var}Func', **kwargs)


class DistributionPlantParameter(OptionPlantParameter):
    r"""Class for a distribution."""

    _name = 'Dist'
    _help = 'Distribution applied to {parent}'
    _properties = {
        'Mean': DelayedPlantParameter(
            'scalar', specialize_name='Mean',
            exclude=['Dist'],
            _help='Mean of {parents[1]} normal distribution',
        ),
        'StdDev': DelayedPlantParameter(
            'scalar', specialize_name='StdDev',
            exclude=['Dist'],
            _help=('Standard deviation of {parents[1]} normal '
                   'distribution'),
        ),
        'Bounds': {
            'type': 'array', 'items': {'type': 'number'},
            'minItems': 2, 'maxItems': 2,
            'description': ('Minimum and maximum of the {parents[0]} '
                            'distribution'),
        },
    }
    _option_dependencies = {
        'normal': ['Mean', 'StdDev'],
        'uniform': ['Bounds'],
    }

    @property
    def profile(self):
        r"""str: Distribution profile."""
        return self.get('')

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        profile = self.profile
        kws = {k: self.get(k, **kwargs) for k in
               self._option_dependencies[profile]}
        kwargs = dict(kws, **kwargs)
        return self.sample_generator_dist(
            self.generator, profile=profile, **kwargs
        )

    @classmethod
    def sample_generator_dist(cls, generator, profile='normal',
                              args=None, **kwargs):
        r"""Sample a distribution using the current random number
        generator.

        Args:
            generator (np.random.Generator): Generator to use.
            profile (str): Name of the profile that should be sampled.
            args (tuple, optional): Positional arguments to pass to the
                generator method. If not provided, the profile parameters
                must be provided by kwargs.
            **kwargs: Additional keyword arguments are checked for the
                required profile parameters if args is not provided. Any
                remaining keyword arguments are passed to the generator
                method.

        """
        if args is None:
            if profile == 'uniform':
                args = kwargs.pop('Bounds')
            else:
                args = [kwargs.pop(k) for k in
                        cls._option_dependencies[profile]]
        return getattr(generator, profile)(*args, **kwargs)

    @classmethod
    def parametrize_dist(cls, values, profile='normal', **kwargs):
        r"""Parameterize a distribution of values.

        Args:
            values (pandas.DataFrame, np.ndarray): Set of values.
            profile (str, optional): Distribution profile that should be
                parameterized.
            **kwargs: Additional keyword arguments are passed to the
                methods used to determine the distribution parameters.

        Returns:
            tuple: Set of parameters for the distribution.

        """
        if isinstance(values, pd.DataFrame):
            values = values.to_numpy()
        if profile in ['normal', 'gauss', 'gaussian']:
            param = {
                'Mean': np.nanmean(values, **kwargs),
                'StdDev': np.nanstd(values, **kwargs),
            }
        # elif profile in ['choice']:
        #     param = (values, )
        elif profile in ['uniform']:
            param = {
                'Bounds': np.nanmin(values, **kwargs),
            }
        else:
            raise ValueError(f"Unsupported profile \"{profile}\"")
        return param


class ColorPlantParameter(SimplePlantParameter):
    r"""Class for color parameters."""

    _name = 'color'
    _unit_dimension = None
    _properties = {
        '': {
            'oneOf': [
                {'type': 'string'},
                {
                    'type': 'array', 'minItems': 3, 'maxItems': 3,
                    'items': {
                        'type': 'integer', 'minimum': 0, 'maximum': 255
                    },
                },
            ],
            # 'default': (0,255,20),
            'default': 'summer',
            'description': ('Matplotlib colormap name or initial RGB '
                            'color tuple for {parent}'),
        },
        'Final': {
            'type': 'array', 'minItems': 3, 'maxItems': 3,
            'items': {'type': 'integer', 'minimum': 0, 'maximum': 255},
            'description': 'Final RGB color tuple for {parent}.'
        },
        'Func': FunctionPlantParameter.specialize(
            'color_func',
            _help=('Function to map between parameter values and the '
                   'distance along the color map or between the initial '
                   'and final colors '),
            _default='linear',
            defaults={
                'VarName': 'Age',
                'VarMin': 'AgeSenesce',
                'VarMax': 'AgeRemove',
                'Slope': 1.0,
                'Intercept': 0.0,
            },
        ),
    }
    _required = ['']

    def __init__(self, *args, **kwargs):
        self._starting_color_idx = None
        super(ColorPlantParameter, self).__init__(*args, **kwargs)

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        if self._starting_color_idx is None:
            self.initialize_lpy_turtle(self.root.context.turtle)
        v = self.get('Func', **kwargs)
        c0 = self.get('')
        c1 = self.get('Final', None)
        out = self._starting_color_idx
        if isinstance(c0, str) or c1 is not None:
            out += int(255 * v)
        return out

    def initialize_lpy_context(self, context):
        r"""Initialize the lpy context for this instance.

        Args:
            context (lpy.LsysContext): Context to initialize.

        """
        if self.initialized_lpy_context:
            return
        super(ColorPlantParameter, self).initialize_lpy_context(context)

    def initialize_lpy_turtle(self, turtle):
        r"""Initialize a PglTurtle instance.

        Args:
            turtle (plantgl.PglTurtle): Turtle to update.

        """
        import openalea.plantgl.all as pgl
        c0 = self.get('')
        c1 = self.get('Final', None)
        cmap = None
        if isinstance(c0, str):
            import matplotlib as mpl
            cmap = mpl.colormaps[c0]
        elif c1 is not None:
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                self.fullname, [c0, c1])
        colors = [c0] if cmap is None else [
            tuple(int(255 * x) for x in cmap(v)[:-1])
            for v in np.linspace(0, 1, 256)
        ]
        self._starting_color_idx = len(turtle.getColorList())
        for i, color in enumerate(colors):
            idx = len(turtle.getColorList())
            name = self.component2colorname(self.component_name, i)
            material = pgl.Material(name, ambient=color)
            material.name = name
            turtle.setMaterial(idx, material)
        super(ColorPlantParameter, self).initialize_lpy_turtle(turtle)

    @classmethod
    def component2colorname(cls, component, idx):
        r"""Create a color name based on the component and color index.

        Args:
            component (str): Name of component.
            idx (int): Color index.

        """
        return f'{component}_{idx}'

    @classmethod
    def colorname2component(cls, name):
        r"""Extract the component name from a color name.

        Args:
            name (str): Color name.

        """
        return name.rsplit('_', 1)[0]


class ScalarPlantParameter(SimplePlantParameter):
    r"""Class for scalar parameters that will have a spread by default
    and can have a dependence on age, n, x, or w."""

    _name = 'scalar'
    _unit_dimension = None
    _properties = {
        '': {
            'type': 'number',
            'description': 'Mean of {parent}',
        },
        'Min': {
            'type': 'number',
            'description': ('Minimum allowed value. Samples of '
                            'distributions that push the value below '
                            'this value will be clipped to the minimum'),
        },
        'Max': {
            'type': 'number',
            'description': ('Maximum allowed value. Samples of '
                            'distributions that push the value above '
                            'this value will be clipped to the maximum'),
        },
        'StdDev': DelayedPlantParameter(
            'scalar', specialize_name='StdDev',
            exclude=['Dist', 'StdDev', 'RelStdDev'],
            _help=('Standard deviation of {parents[1]} normal '
                   'distribution'),
            # properties={
            #     'VarName': {
            #         'type': 'array', 'items': {'type': 'string'},
            #         'description': ('Names of index variables that the '
            #                         'variance should be dependent on'),
            #     },
            # },
            # defaults={
            #     'VarName': ['N'],
            # },
        ),
        'RelStdDev': DelayedPlantParameter(
            'scalar', specialize_name='RelStdDev',
            exclude=['Dist', 'StdDev', 'RelStdDev'],
            _help=('Standard deviation of {parents[1]} normal '
                   'distribution relative to the mean'),
        ),
        'Dist': DistributionPlantParameter.specialize(
            'scalar_dist',
            _help='Distribution that {parents[0]} should be scaled by',
            _dependencies=['N'],
            # ['X', 'N', 'Age'],  # Force update for each set
            exclude_options=['normal'],
        ),
        'Func': FunctionPlantParameter.specialize(
            'scalar_func',
            _help='Function that {parents[0]} should be scaled by',
        ),
        'XFunc': FunctionPlantParameter.specialize_var(
            'X',
            _help=('Dependence of {parents[0]} on the distance along the '
                   '{component}'),
        ),
        'NFunc': FunctionPlantParameter.specialize_var(
            'N', normvar='NMax',
            component_properties=['NMax'],
            _help=('Dependence of {parents[0]} on the {component} '
                   'phytomer number'),
        ),
        'AgeFunc': FunctionPlantParameter.specialize_var(
            'Age', normvar='AgeMature', maxvar='AgeMature',
            component_properties=['AgeMature'],
            _help=('Dependence of {parents[0]} on the {component} age'),
            properties={
                'AgeMature': {
                    'type': 'scalar', 'subtype': 'float', 'units': 'days',
                    'description': 'Age after which {component} matures',
                },
            },
        ),
        'WFunc': FunctionPlantParameter.specialize_var(
            'W', normvar='WMax',
            component_properties=['WMax'],
            _help=('Dependence of {parents[0]} on the {component} '
                   'internal iterator count'),
        ),
        'BFunc': FunctionPlantParameter.specialize_var(
            'B', normvar='BMax',
            component_properties=['BMax'],
            _help=('Dependence of {parents[0]} on the {component} '
                   'branch level'),
        ),
    }
    _required = ['']
    _modifiers = [
        'Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc', 'WFunc', 'BFunc',
    ]
    _property_dependencies = dict(
        SimplePlantParameter._property_dependencies,
        **{k: {'': True} for k in [
            'Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc', 'WFunc',
            'BFunc',
        ]}
    )
    _property_dependencies_defaults = dict(
        SimplePlantParameter._property_dependencies_defaults,
        **{'': {
            'value': 1.0,
            'conditions': {
                k: True for k in [
                    'Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc', 'WFunc',
                    'BFunc',
                ]
            },
        }}
    )

    @cached_property
    def index_parameters(self):
        r"""list: Set of index parameters that this parameter uses."""
        out = copy.deepcopy(
            super(ScalarPlantParameter, self).index_parameters)
        # TODO: Allow this to be specified
        if 'StdDev' in self.parameters or 'RelStdDev' in self.parameters:
            # Force update for each set
            out += [
                k for k in ['N'] if k not in out
            ]
        return out

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        base = self.get('', 1.0)
        self.log(f'base = {base}')
        stddev = self.get('StdDev', None)
        vmin = self.get('Min', None)
        vmax = self.get('Max', None)
        out = base
        for k in self._modifiers:
            v = self.get(k, None)
            if v is not None:
                self.log(f'{k} = {v}')
                out *= v
        if stddev is None:
            relstddev = self.get('RelStdDev', None)
            if relstddev is not None:
                stddev = np.abs(relstddev * out)
        if stddev is not None:
            self.log(f"SAMPLE STDDEV: {out}, {stddev}")
            try:
                out = self.sample_dist('normal', Mean=out, StdDev=stddev)
            except BaseException:
                self.log(f"SAMPLE STDDEV: {out}, {stddev}", force=True)
                raise
        if vmin is not None and out < vmin:
            out = vmin
        if vmax is not None and out > vmax:
            out = vmax
        return out


class CurvePlantParameter(SimplePlantParameter):
    r"""Class for a curve parameter.

    ClassAttributes:
        _required_curve (list): Properties required when the curve is
            not generated from a curve patch.
        _required_patch (list): Properties required when the curve is
            generated from a curve patch.
        _allow_patch (bool): True if curve patches should be allowed.

    """

    _name = 'curve'
    _properties = {
        'ControlPoints': {
            'type': 'ndarray', 'subtype': 'float', 'ndim': 2,
            'description': '2D control points for the {parent} curve'
        },
        'Symmetry': {
            'type': 'array', 'items': {'type': 'integer'},
            'description': 'Dimensions of the {parent} curve symmetry',
        },
        'Closed': {
            'type': 'boolean', 'default': False,
            'description': 'Should the the {parent} curve be closed',
        },
        'Reverse': {
            'type': 'boolean', 'default': False,
            'description': ('Should the {parent} control points be '
                            'added to the curve in reverse order'),
        },
        'Thickness': {
            'type': 'number',
            'description': ('Double the {parent} curve back on itself '
                            'with this thickeness'),
        },
        'Patch': DelayedPlantParameter(
            'curve_patch',
            _help=('Curve patch that should be sampled to generate the '
                   '{parent} curve'),
        ),
        'PatchVarName': {
            'type': 'string',
            'description': ('Parameter that should be used to sample '
                            'the {parent}Patch'),
        },
        'PatchNorm': {
            'type': 'number',
            'description': ('Value that should be used to normalize '
                            '{parent}PatchVarName prior to '
                            'sampling {parent}Patch'),
        },
        'PatchMin': {
            'type': 'number',
            'description': ('Minimum value over which '
                            '{parent}PatchVarName is valid'),
        },
        'PatchMax': {
            'type': 'number',
            'description': ('Maximum value over which '
                            '{parent}PatchVarName is valid'),
        },
    }
    _property_dependencies = {
        'Symmetry': {'ControlPoints': True},
        'Closed': {'ControlPoints': True},
        'Reverse': {'ControlPoints': True},
        'Thickness': {'ControlPoints': True},
        'PatchVarName': {'Patch': True},
        'PatchNorm': {'Patch': True, 'PatchVarName': True},
        'PatchMin': {'Patch': True, 'PatchVarName': True},
        'PatchMax': {'Patch': True, 'PatchVarName': True},
    }
    _required = []
    _required_curve = ['ControlPoints']
    _required_patch = ['Patch', 'PatchVarName']
    _allow_patch = True

    @staticmethod
    def _on_registration(cls):
        SimplePlantParameter._on_registration(cls)
        cls._patch_properties = [k for k in cls._properties.keys()
                                 if k.startswith('Patch')]
        cls._curve_properties = cls._required_curve
        cls._allow_patch = ('Patch' in cls._properties)
        if cls._allow_patch:
            cls._shared = [
                k for k in cls._properties.keys()
                if k not in
                cls._patch_properties + cls._curve_properties
            ]
            cls._variables = [
                k for k in cls._variables
                if k not in cls._curve_properties
            ] + ['PatchVarName']

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        out = super(CurvePlantParameter, cls).schema()
        if not cls._allow_patch:
            out['required'] = copy.deepcopy(cls._required_curve)
            return out
        out['oneOf'] = []
        for opt in ['curve', 'patch']:
            optreq = getattr(cls, f'_required_{opt}')
            out['oneOf'].append({
                'type': 'object',
                'properties': {
                    k: copy.deepcopy(out['properties'][k])
                    for k in optreq
                },
                'required': copy.deepcopy(optreq),
            })
            for k in optreq:
                out['properties'].pop(k)
        return out

    @classmethod
    def add_class_defaults(cls, param, **kwargs):
        r"""Update defaults based on parameters.

        Args:
            param (ParameterDict): Parameter set to update.
            **kwargs: Additional keyword arguments are added to param.

        """
        super(CurvePlantParameter, cls).add_class_defaults(
            param, **kwargs)
        # TODO: Verify that curve properties arn't remove due to
        #   property_dependencies in parent method
        if cls._allow_patch and not all(k in param for k in
                                        cls._required_curve):
            for k in cls._shared:
                v = param.get(k, NoDefault)
                if v is NoDefault:
                    continue
                for kchild in ['Start', 'End']:
                    param.setdefault(f'Patch{kchild}Curve{k}', v)

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        if self._allow_patch and 'Patch' in self.parameters:
            if kwargs:
                patch = self.get('Patch', return_other='instance')
                return (
                    patch.get('StartCurve', **kwargs),
                    patch.get('EndCurve', **kwargs),
                )
            patch = self.get('Patch')
            tvar = self.get('PatchVarName')
            tnorm = self.get('PatchNorm', None)
            tmin = self.get('PatchMin', None)
            tmax = self.get('PatchMax', None)
            t = self.get(tvar)
            curve = CurvePatchPlantParameter.sample_curve_patch(
                patch, t, tnorm=tnorm, tmin=tmin, tmax=tmax)
        else:
            points = self.get('ControlPoints')
            kwargs.setdefault('symmetry', self.get('Symmetry', None))
            kwargs.setdefault('closed', self.get('Closed', False))
            kwargs.setdefault('reverse', self.get('Reverse', False))
            kwargs.setdefault('thickness', self.get('Thickness', None))
            curve = self.create_curve(points, **kwargs)
        return curve

    @classmethod
    def reflect_points(cls, points_R, axis):
        r"""Reflect points across an axis.

        Args:
            points_R (np.ndarray): Points that should be reflected from
                the positive side of the specified axis to the negative
                side.
            axis (int): Index of axis that points should be reflected
                across.

        Returns:
            np.ndarray: Set of points including the original points and
                the reflected points.

        """
        scale = np.ones((points_R.shape[1], ))
        scale[axis] = -1.0
        points_L = points_R[::-1] * scale
        if np.allclose(points_L[-1, :], points_R[0, :]):
            points_L = points_L[:-1, :]
        return np.concatenate([points_L, points_R])

    @classmethod
    def sample_curve(cls, curve, t):
        r"""Sample a curve.

        Args:
            curve (NurbsCurve): Curve to sample.
            t (float): Value to sample the curve at.

        Returns:
            float: Curve value at value t.

        """
        raise NotImplementedError("Sampling curve")

    @classmethod
    def normalize_points(cls, points, closed=False):
        r"""Normalize a set of points so that the maximum length of the
        curve (if open) or area inside the polygon (if closed) is 1.

        Args:
            points (np.ndarray): Points to scale.
            closed (bool, optional): If True, the curve is closed.

        Returns:
            np.ndarray: Scaled points.

        """
        if closed:
            diff = points - points[:, None]
            dist = np.max(np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)))
        else:
            diff = points[1:, :] - points[:-1, :]
            dist = np.sum(np.sqrt(diff ** 2))
        return points / dist

    @classmethod
    def create_curve(cls, points, knots=None, uniform=False,
                     stride=60, degree=3, symmetry=None, closed=False,
                     reverse=False, thickness=None, factor=None,
                     return_points=False, return_area=False):
        r"""Create a PlantGL NURBS curve.

        Args:
            points (np.ndarray): Control points. If points are 2D a
                NurbsCurve2D instance will be created, otherwise a
                NurbsCurve instance will be created.
            knots (list, optional): Knot list.
            uniform (bool, optional): If True, the spacing between control
                points is made to be uniform.
            stride (int, optional): The curve stride.
            degree (int, optional): The curve degree.
            symmetry (list, optional): The indices of axes that the points
                should be reflected over.
            closed (bool, optional): If True, the curve is closed.
            reverse (bool, optional): If True, the order of the points
                should be reversed.
            thickness (float, optional): Thickness that should be added
                to the curve by doubling it back on itself.
            factor (float, optional): Additional scale factor that should
                be applied to points.
            return_points (bool, optional): If True, the control points
                for the curve will be returned instead of the curve.
            return_area (bool, optional): If True, return the area
                contained by the curve.

        Returns:
            NurbsCurve: Curve instance.

        """
        if factor is not None:
            points = factor * points
        if symmetry is not None:
            for ax in symmetry:
                points = cls.reflect_points(points, ax)
        if reverse:
            points = points[::-1, :]
        area = None
        if thickness is not None:
            assert thickness < 1
            closed = True
            if points.shape[1] != 2:
                raise NotImplementedError('Thickness for 3D curve')
            points = cls.normalize_points(points)
            nthickness1 = int(np.ceil(points.shape[0] / 2))
            nthickness2 = int(np.floor(points.shape[0] / 2))
            assert (nthickness1 + nthickness2) == points.shape[0]
            thickness1 = np.linspace(0, thickness, nthickness1)
            thickness2 = thickness1[:nthickness2]
            thickness0 = np.concatenate([thickness1, thickness2[::-1]])
            points_bottom = points.copy()
            points_bottom[:, 1] -= thickness0
            if return_area:
                area_top = np.prod(points[1:, :] - points[:-1, :]) / 2
                area_bot = np.prod(points_bottom[1:, :]
                                   - points_bottom[:-1, :]) / 2
                out = area_top - area_bot
                assert len(out) == len(thickness0 - 1)
                area = sum(out)
            points_bottom = points_bottom[::-1, :]
            points = np.concatenate([
                points[:1, :], points,
                points_bottom, points_bottom[-1:, :],
            ])
        if closed and points.shape[1] == 2:
            idx0 = np.argmax(points[:, 1])
            points = np.concatenate([
                points[idx0:, :], points[:idx0, :]
            ])
        if closed and not np.allclose(points[0, :], points[-1, :]):
            points = np.concatenate([points, points[:1, :]])
        if thickness is None:
            points = cls.normalize_points(points, closed=closed)
        if return_area:
            assert closed
            out = 0.5 * np.abs(np.dot(points[:, 0],
                                      np.roll(points[:, 1], 1))
                               - np.dot(points[:, 1],
                                        np.roll(points[:, 0], 1)))
            if area is not None:
                print("COMPARE CALCS")
                print(area)
                print(out)
                pdb.set_trace()
                out = area
            return out
        if return_points:
            return points
        use2D = False
        if points.shape[1] == 2:
            use2D = True
            ctrl_points = [Vector2(*vec) for vec in points]
        else:
            ctrl_points = [Vector3(*vec) for vec in points]
        nb_pts = len(ctrl_points)
        nb_arc = (nb_pts - 1) // degree
        # nb_knots = degree + nb_pts
        p = 0.
        param = [p]
        for i in range(nb_arc):
            if uniform:
                p += 1
            else:
                p += pglmath.norm(
                    ctrl_points[degree * i]
                    - ctrl_points[degree * (i + 1)]
                )
            param.append(p)
        if knots is None:
            knots = [param[0]]
            for p in param:
                for j in range(degree):
                    knots.append(p)
            knots.append(param[-1])
        if use2D:
            out = NurbsCurve2D(
                [Vector3(v[0], v[1], 1.) for v in ctrl_points],
                knots, degree, stride,
            )
        else:
            out = NurbsCurve(
                [Vector4(v[0], v[1], v[2], 1.) for v in ctrl_points],
                knots, degree, stride,
            )
        return out


class CurvePatchPlantParameter(SimplePlantParameter):
    r"""Class for a curve patch parameter."""

    _name = 'curve_patch'
    _help = 'Set of curves transitioning from one curve to another'
    _properties = {
        'Start': CurvePlantParameter.specialize(
            'start_curve', exclude=CurvePlantParameter._patch_properties,
            _help='Curve defining the start of the patch',
        ),
        'End': CurvePlantParameter.specialize(
            'end_curve', exclude=CurvePlantParameter._patch_properties,
            _help='Curve defining the end of the patch',
        ),
    }
    _required = ['Start', 'End']

    @classmethod
    def sample_curve_patch(cls, p, t, tnorm=None, tmin=None, tmax=None):
        r"""Sample a curve patch.

        Args:
            p (NurbsPatch): Curve patch to sample.
            t (float): Value to sample the patch at.
            tnorm (float, optional): Value to normalize t against.
            tmin (float, optional): Value before which the first patch
                should be used.
            tmax (float, optional): Value after which the last patch
                should be used.

        Returns:
            NurbsCurve2D: Curve corresponding to value t.

        """
        if tnorm is not None:
            t = t / tnorm
            if tmin is not None:
                tmin = tmin / tnorm
            if tmax is not None:
                tmax = tmax / tnorm
        if tmin is not None and t < tmin:
            t = 0.0
        elif tmax is not None and t > tmax:
            t = 1.0
        else:
            if tmin is None:
                tmin = 0.0
            if tmax is None:
                tmax = 1.0
            t = (t - tmin) / (tmax - tmin)
        section = p.getIsoUSectionAt(t)
        pts = [(i.x, i.y, i.w) for i in section.ctrlPointList]
        return NurbsCurve2D(pts, section.knotList, section.degree)

    @classmethod
    def create_curve_patch(cls, curves, knots=None, degree=3):
        r"""Create a PlantGL NURBS curve patch.

        Args:
            curves (list): Set of NurbsCurve instances defining the
                patch.
            knots (list, optional): Knot list.
            degree (int, optional): The curve degree.

        Returns:
            NurbsPatch: Curve instance.

        """
        # nbcurves = len(curves)
        # if knots is None:
        #     knots = [i / float(nbcurves - 1) for i in range(nbcurves)]
        # k = [
        #     knots[0] for i in range(degree - 1)
        # ] + knots + [
        #     knots[-1] for i in range(degree - 1)
        # ]
        pts = [
            [(i.x, i.y, 0, 1) for i in c.ctrlPointList] for c in curves
        ]
        ppts = pgl.Point4Matrix(pts)
        return NurbsPatch(ppts, udegree=degree, vdegree=curves[0].degree)

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        start = self.get('Start', **kwargs)
        end = self.get('End', **kwargs)
        return self.create_curve_patch([start, end])


class Template1DPlantParameter(SimplePlantParameter):
    r"""Class for loading and scaling a 1D template from a shape file."""

    _name = 'Template1D'
    _help = '1D {component} template loaded from file'
    _ndim = 1
    _properties = {
        '': {
            'type': 'string',
            # 'help': 'File containing a 1D {component} template',
        },
        'Index': {
            'type': 'integer',
            'help': ('Index of the shape that should be used from the '
                     '{parent} file'),
            'default': 0,
        },
        'XScale': ScalarPlantParameter.specialize(
            'x_scale',
            _help='Scale factor for {parent} along the x direction',
        ),
    }

    def _generate(self, **kwargs):
        r"""Generate this parameter.

        Args:
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        import shapefile
        shp = shapefile.Reader(self.get('')).shapes()[self.get('Index')]
        pts = np.array(shp.points)
        assert pts.ndim == 2
        assert pts.shape[-1] == self._ndim
        for i, x in enumerate('XYZ'):
            if i >= self._ndim:
                break
            pts[:, i] *= self.get(f'{x}Scale', **kwargs)
        return pts


class Template2DPlantParameter(Template1DPlantParameter):
    r"""Class for loading and scaling a 2D template from a shape file."""

    _name = 'Template2D'
    _help = '2D {component} template loaded from file'
    _ndim = 2
    _properties = dict(
        Template1DPlantParameter._properties,
        YScale=ScalarPlantParameter.specialize(
            'y_scale',
            _help='Scale factor for {parent} along the y direction',
        ),
    )


class Template3DPlantParameter(Template2DPlantParameter):
    r"""Class for loading and scaling a 3D template from a shape file."""

    _name = 'Template3D'
    _help = '3D {component} template loaded from file'
    _ndim = 3
    _properties = dict(
        Template2DPlantParameter._properties,
        ZScale=ScalarPlantParameter.specialize(
            'z_scale',
            _help='Scale factor for {parent} along the y direction',
        ),
    )


class ParameterCollection(SimplePlantParameter):
    r"""Class for collection of parameters.

    ClassAttributes:
        _parameters (dict): Mapping between parameter class names and
            lists of parameter names that should be included in this
            collection using that class.

    """

    _name = 'collection'
    _parameters = {}
    _properties = {}

    @staticmethod
    def _on_registration(cls):
        SimplePlantParameter._on_registration(cls)
        cls._properties = copy.deepcopy(cls._properties)
        for k, v in cls._parameters.items():
            kcls = get_class_registry().get('plant_parameter', k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    cls._properties[kk] = kcls.specialize(kk, **vv)
            else:
                for kk in v:
                    cls._properties[kk] = kcls


###############################################
# COMPONENTS
###############################################


class ComponentBase(SimplePlantParameter):
    r"""Base class for generating plant components."""

    _registry_key = 'plant_component'
    _name = None
    _name_plural = None
    _help = None
    _subcomponents = []
    _properties = dict(
        ParameterCollection._properties,
        WMax={
            'type': 'integer',
            'description': 'Number of components in a whorl',
        },
        NFirst={
            'type': 'integer',
            'description': 'First node where component is produced',
        },
        NLast={
            'type': 'integer',
            'description': 'Last node where component is produced',
        },
        NPeriod={
            'type': 'integer',
            'description': 'Number of nodes between each component',
        },
        BFirst={
            'type': 'integer',
            'description': 'First branch level where component is produced',
        },
        BLast={
            'type': 'integer',
            'description': 'Last branch level where component is produced',
        },
        BPeriod={
            'type': 'integer',
            'description': 'Number of branch levels between each component',
        },
    )
    _defaults = dict(
        SimplePlantParameter._defaults,
        WMax=1,
        NFirst=0,
        NLast=-1,
        NPeriod=1,
        BFirst=0,
        BLast=-1,
        BPeriod=1,
    )

    @staticmethod
    def _on_registration(cls):
        if cls._name:
            cls._help = cls._name.lower()
            if not cls._name_plural:
                cls._name_plural = cls._name + 's'
        SimplePlantParameter._on_registration(cls)

    @property
    def component(self):
        r"""str: Component that this property belongs to."""
        return self

    def production_check_remove(self, production=False):
        r"""list: Lsystem production rule lines to remove component if
        n or b are outside the limits for the component."""
        out = []
        for k in ['n', 'b']:
            out += [
                f'{k}min = generator.{self.name}{k.upper()}First()',
                f'{k}max = generator.{self.name}{k.upper()}Last()',
                f'{k}period = generator.{self.name}{k.upper()}Period()',
                f'if (({k} < {k}min',
                f'     or ({k}max != -1 and {k} > {k}max)',
                f'     or ((({k} - {k}min) % {k}period) != 0))):',
            ]
            if production:
                out += [
                    f'    print(f"Removing {self.name}[{k}={{{k}}}] P")',
                    '    produce *',
                ]
            else:
                out += [
                    f'    print(f"Removing {self.name}[{k}={{{k}}}]")',
                    '    produce [/(0)]',
                ]
        return out

    @property
    def additional_production_conditions(self):
        r"""list: Lsystem production rule lines for before 'else'."""
        return []

    @property
    def production(self):
        r"""list: Lsystem production rule lines."""
        args = self.lsystem_args(for_rule=True)
        args_inc = self.lsystem_args(ageinc=True, for_rule=True)
        prefix = f'generator.{self.name}'
        out = self.production_check_remove(production=True)
        out += [
            f'if age >= {prefix}AgeRemove():',
            '    if generator.verbose_lpy:',
            f'        print(f"Removing {self.name}[{{age}},{{n}}]")',
            '    produce *',
            'elif (time + age) >= OUTPUT_TIME:',
            f'    produce {self.name}({args})',
        ]
        out += self.additional_production_conditions
        out += [
            'else:',
            '    if generator.verbose_lpy:',
            f'        print(f"Ageing {self.name}[{{age}},{{n}}]")',
            '    ageinc = min(generator.AgeInc(),',
            '                 OUTPUT_TIME - (time + age))',
            f'    produce {self.name}({args_inc})',
        ]
        return out

    @property
    def homomorphism(self):
        r"""list: Lsystem homomorphism rule lines."""
        return ['produce &(90)@o^(90)']


class GeometricComponentBase(ComponentBase, OptionPlantParameter):
    r"""Base class for generating plant component geometries."""

    _name_option = 'Method'
    _properties = dict(
        ComponentBase._properties,
        Method={
            'type': 'string',
            'description': ('Method that should be used to generate '
                            'each {component}'),
        },
        Length=ScalarPlantParameter.specialize(
            'length',
            _help='{Component} length',
            _unit_dimension='length',
            constants={
                'Min': 0,
            },
            defaults={
                'AgeFunc': 'linear',
                'AgeFuncSlope': 1.0,
                'AgeFuncIntercept': 0.0,
            },
        ),
        Width=ScalarPlantParameter.specialize(
            'width',
            _help='{Component} width',
            _unit_dimension='length',
            constants={
                'Min': 0,
            },
            defaults={
                'AgeFunc': 'linear',
                'AgeFuncSlope': 1.0,
                'AgeFuncIntercept': 0.0,
            },
        ),
        Angle=ScalarPlantParameter.specialize(
            'polar_angle',
            _help='{Component} angle from parent forward axis',
            _defaults={'': 0},
            _unit_dimension='angle',
        ),
        RotationAngle=ScalarPlantParameter.specialize(
            'polar_angle',
            _help='{Component} rotation angle around parent forward axis',
            _defaults={'': 0},
            _unit_dimension='angle',
        ),
        Bend=ScalarPlantParameter.specialize(
            'bend',
            _help=('Angle that {component} bends per distance along its '
                   'length'),
            _unit_dimension='angle/length',
            defaults={
                '': 0,
            }
        ),
        Twist=ScalarPlantParameter.specialize(
            'twist',
            _help=('Angle that {component} twists per distance along '
                   'its length'),
            _unit_dimension='angle/length',
            defaults={
                '': 0,
            }
        ),
        NDivide={
            'type': 'integer', 'default': 10,
            'description': ('The number of segments that {component} '
                            'will be dividied into'),
        },
        Thickness=ScalarPlantParameter.specialize(
            'thickness',
            _help='Thickness of each {component}',
            _unit_dimension='length',
        ),
        Profile=CurvePlantParameter.specialize(
            'profile',
            _help='Profile of {component} cross section',
            component_properties=['Thickness'],
        ),
        Template2D=Template2DPlantParameter,
        Template3D=Template3DPlantParameter,
        Color=ColorPlantParameter.specialize(
            'component_color',
            _help='{Component} color',
        ),
        Density={
            'type': 'scalar', 'subtype': 'float', 'units': 'g/cm**3',
            'description': ('Density of the component tissue that '
                            'should be used to calculate the component '
                            'mass based on its volume.'),
        },
        # AvailableCarbon={
        #     'type': 'scalar', 'subtype': 'float', 'units': 'g',
        #     'description': (
        # },
    )
    _defaults = dict(
        OptionPlantParameter._defaults,
        **ComponentBase._defaults
    )
    _option_dependencies = {
        'cylinder': [
            'NDivide',
        ],
        'sweep': [
            'Profile', 'Thickness', 'NDivide',
        ],
        'sphere': [],
        'template2d': ['Template2D', 'Thickness'],
        'template3d': ['Template3D'],
        # 'compound': [],
        'nongeometric': [],
    }

    @staticmethod
    def _on_registration(cls):
        if cls._name:
            cls._help = cls._name.lower()
            if not cls._name_plural:
                cls._name_plural = cls._name + 's'
        OptionPlantParameter._on_registration(cls)

    @cached_property
    def index_parameters(self):
        r"""list: Set of index parameters that this parameter uses."""
        out = copy.deepcopy(super(GeometricComponentBase, self).index_parameters)
        if 'NDivide' in self.parameters and 'X' not in out:
            out.append('X')
        return out

    def XIterator(self):
        r"""Iterator over X values."""
        assert 'X' in self.index_parameters
        DX = 1.0 / self.get("NDivide")
        assert self.current_index['x'] is None
        try:
            x = 0
            while x <= 1:
                self.current_index['x'] = x
                yield x
                if x == 1:
                    break
                x = min(x + DX, 1)
        finally:
            self.current_index['x'] = None

    def Volume(self):
        r"""units.Quantity: The volume contained within the component."""
        out = None
        method = self.get('Method')
        if method in ['cylinder', 'sweep']:
            if 'NDivide' in self.parameters:
                length = self.get('Length')
                out = []
                xprev = 0.0
                for x in self.XIterator():
                    r = self.get('Width') / 2
                    if 'Profile' in self.parameters:
                        area = self.get('Profile', return_area=True)
                    else:
                        area = np.pi
                    out.append(length * (x - xprev) * area * (r ** 2))
                    xprev = x
                return sum(out)
            else:
                length = self.get('Length')
                r = self.get('Width') / 2
                if 'Profile' in self.parameters:
                    area = self.get('Profile', return_area=True)
                else:
                    area = np.pi
                return length * area * (r ** 2)
        else:
            raise NotImplementedError(method)

    def produce_geometry(self, for_rule=False):
        r"""Get the LSystem lines for producing the geometry.

        Args:
            for_rule (bool, optional): If True, format for a rule.

        Returns:
            list: Lsystem lines to produce the component geometry.

        """
        args = self.lsystem_args(for_rule=for_rule)
        prefix = f'generator.{self.name}'
        out = []
        if 'Color' in self.parameters:
            out += [
                f'nproduce SetColor({prefix}Color({args}))',
            ]
        if 'RotationAngle' in self.parameters:
            out += [
                f'nproduce /({prefix}RotationAngle({args}))',
            ]
        if 'Angle' in self.parameters:
            out += [
                f'nproduce &({prefix}Angle({args}))',
            ]
        for k in self._subcomponents:
            if k not in self.root.parameters:
                continue
            kargs = self.root.parameters[k].lsystem_args(for_rule=True)
            kout = [
                f'nproduce {k}({kargs})'
            ]
            if 'Profile' in self.parameters:
                kout[0] += ' @Ge @Gc'
            out += kout
        method = self.get('Method')
        if method == 'nongeometric':
            pass
        elif method in ['sphere']:
            out += [
                # TODO: Width unused
                # f'nproduce _({prefix}Width({args}))',
                f'length = {prefix}Length({args})',
                'produce @O(length)',
            ]
        elif method in ['cylinder', 'sweep']:
            if 'Profile' in self.parameters:
                out.append(
                    f'nproduce SetContour({prefix}Profile({args},x=0))'
                )
            out += [
                f'length = {prefix}Length({args})',
                'nproduce F(0.001 * length)',
            ]
            if 'NDivide' in self.parameters:
                args_x = f'{args},x=x'
                out += [
                    'xprev = 0.0',
                    'for x in component.XIterator():',
                ]
                out_x = [
                    'DX = length * (x - xprev)',
                    'xprev = x',
                    f'nproduce _({prefix}Width({args_x}))'
                ]
                if 'Bend' in self.parameters:
                    out_x.append(
                        f'nproduce &({prefix}Bend({args_x})*DX)'
                    )
                if 'Twist' in self.parameters:
                    out_x.append(
                        f'nproduce /({prefix}Twist({args_x})*DX)'
                    )
                if 'Profile' in self.parameters:
                    out_x.append(
                        f'nproduce SetContour({prefix}Profile({args_x}))'
                    )
                out_x += [
                    'if x == 1:',
                    '    produce F(DX)',
                    'else:',
                    '    nproduce F(DX)',
                ]
                out += ['    ' + x for x in out_x]
            else:
                if 'Profile' in self.parameters:
                    out.append(
                        f'nproduce SetContour({prefix}Profile({args}))'
                    )
                out += [
                    f'nproduce _({prefix}Width({args}))',
                    'produce F(length)',
                ]
        else:
            raise NotImplementedError(method)
        return out

    @property
    def homomorphism(self):
        r"""list: Lsystem homomorphism rule lines."""
        args = self.lsystem_args()
        out = self.production_check_remove()
        out += [
            'if generator.verbose_lpy:',
            f'    print(f"Homomorphism {self.name}[{{age}},{{n}}]")',
        ] + self.produce_geometry()
        out = [
            f'component = generator.parameters[\"{self.name}\"]',
            f'with component.temporary_index({args}):'
        ] + ['    ' + x for x in out]
        return out


class PetioleComponent(GeometricComponentBase):
    r"""Petiole component."""

    _name = 'Petiole'
    _properties = {
        k: v for k, v in GeometricComponentBase._properties.items()
        if k not in ['Angle', 'RotationAngle']
    }
    _defaults = dict(
        GeometricComponentBase._defaults,
        Method='cylinder',
    )


class WhorlComponent(SimplePlantParameter):

    _registry_key = 'plant_component'
    _name = 'Whorl'
    _properties = {
        'Elements': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
            'description': 'Component(s) that is repeated.'
        },
    }
    _required = ['Elements']

    @property
    def component(self):
        r"""str: Component that this property belongs to."""
        return self

    @property
    def production(self):
        r"""list: Lsystem production rule lines."""
        elements = [x for x in self.get('Elements')
                    if x in self.root._components]
        out = []
        for element in elements:
            args = self.root.parameters[element].lsystem_args(for_rule=True)
            out += [
                f'for w in range(generator.{element}WMax()):',
                f'    nproduce [{element}({args})]',
            ]
        out += ['produce [/(0)]']
        return out

    @property
    def homomorphism(self):
        r"""list: Lsystem homomorphism rule lines."""
        return []


class NodeComponent(WhorlComponent):
    r"""Node component."""

    _name = 'Node'
    _properties = {}
    _required = []
    _constants = {'Elements': ['Cotyledon', 'Leaf', 'Branch']}


class LeafComponent(GeometricComponentBase):
    r"""Leaf component."""

    _name = 'Leaf'
    _name_plural = 'Leaves'
    _properties = dict(
        GeometricComponentBase._properties,
        Unfurled={
            'type': 'boolean', 'default': False,
            'description': ('The leaf should be unfurled, starting as a '
                            'cylinder and ending as the specified '
                            'LeafProfile'),
        },
        UnfurledLength={
            'type': 'number', 'default': 0.5,
            'description': ('The fraction of the leaf that should be '
                            'unfurled'),
        },
    )
    _subcomponents = ['Petiole']
    _option_dependencies = dict(
        GeometricComponentBase._option_dependencies,
        sweep=GeometricComponentBase._option_dependencies['sweep'] + [
            'Unfurled', 'UnfurledLength',
        ],
    )
    _property_dependencies = dict(
        GeometricComponentBase._property_dependencies,
        UnfurledLength={'Unfurled': [True]},
    )
    _defaults = dict(
        GeometricComponentBase._defaults,
        NFirst=1,  # After cotyledons
        ProfileClosed=False,
        ProfileSymmetry=[0],
        ProfileReverse=True,
        ProfileControlPoints=np.array([
            [+0.0,  0.0],
            [+0.1,  0.0],
            [+0.2,  0.0],
            [+0.5,  0.1],
            [+1.0,  0.2],
        ]),
    )


class CotyledonComponent(LeafComponent):
    r"""Cotyledon component."""

    _name = 'Cotyledon'
    _name_plural = None
    _defaults = dict(
        LeafComponent._defaults,
        NFirst=0,
        NLast=0,
        WMax=2,
    )


class InternodeComponent(GeometricComponentBase):
    r"""Internode component."""

    _name = 'Internode'
    _defaults = dict(
        GeometricComponentBase._defaults,
        Method='cylinder',
        NDivide=10,
        WidthXFunc='linear',
        WidthXFuncIntercept=1.0,
        WidthXFuncSlope=1.0,
        WidthXFuncSlopeAgeFunc='linear',
        WidthXFuncSlopeAgeFuncIntercept=-1.0,
        WidthXFuncSlopeAgeFuncSlope=1.0,
    )


class BranchComponent(ComponentBase):
    r"""Branch component."""

    _name = 'Branch'
    _name_plural = 'Branches'
    _properties = dict(
        ComponentBase._properties,
        Angle=ScalarPlantParameter.specialize(
            'polar_angle',
            _help='{Component} angle from parent forward axis',
            _defaults={'': 0},
            _unit_dimension='angle',
        ),
        RotationAngle=ScalarPlantParameter.specialize(
            'polar_angle',
            _help='{Component} rotation angle around parent forward axis',
            _defaults={'': 0},
            _unit_dimension='angle',
        ),
    )

    @property
    def additional_production_conditions(self):
        r"""list: Lsystem production rule lines for before 'else'."""
        out = super(BranchComponent, self).additional_production_conditions
        prefix = f'generator.{self.name}'
        args = self.lsystem_args()
        args_intn = self.root.parameters['Internode'].lsystem_args(
            for_rule=True, ninc=1, binc=1)
        args_node = self.root.parameters['Node'].lsystem_args(
            for_rule=True, ninc=1, binc=1)
        out += [
            'elif age >= PLASTOCHRON and n < NMax:',
            '    time = time + age',
            '    if age >= PLASTOCHRON:'
            '        age = age - PLASTOCHRON',
            '    if generator.verbose_lpy:',
            '        print(f"Auxillary node: {n+1} (t = {time})")',
        ]
        if 'RotationAngle' in self.parameters:
            out += [
                f'    nproduce /({prefix}RotationAngle({args}))',
            ]
        if 'Angle' in self.parameters:
            out += [
                f'    nproduce &({prefix}Angle({args}))',
            ]
        out += [
            f'    nproduce @Ge @Gc Internode({args_intn})',
            f'    nproduce [Node({args_node})]',
        ]
        return out


class ApexComponent(ComponentBase):
    r"""Apex component."""

    _name = 'Apex'
    _name_plural = 'Apexes'

    @property
    def additional_production_conditions(self):
        r"""list: Lsystem production rule lines for before 'else'."""
        out = super(ApexComponent, self).additional_production_conditions
        args_next = self.lsystem_args(for_rule=True, ninc=1)
        args_intn = self.root.parameters['Internode'].lsystem_args(
            for_rule=True)
        args_node = self.root.parameters['Node'].lsystem_args(
            for_rule=True)
        out += [
            'elif ((age >= PLASTOCHRON and n < NMax)',
            '      or ((time + age) == 0 and n == 0)):',
            '    time = time + age',
            '    if age >= PLASTOCHRON:'
            '        age = age - PLASTOCHRON',
            '    if generator.verbose_lpy:',
            '        print(f"Node: {n} (t = {time})")',
            f'    nproduce @Ge @Gc Internode({args_intn})',
            f'    nproduce [Node({args_node})]',
            '    if generator.verbose_lpy:',
            '        print(f"Next apex: {n + 1}")',
            f'    produce Apex({args_next})',
        ]
        return out


class BudComponent(GeometricComponentBase):
    r"""Bud component."""

    _name = 'Bud'


class PedicelComponent(GeometricComponentBase):
    r"""Pedicel component."""

    _name = 'Pedicel'
    _properties = {
        k: v for k, v in GeometricComponentBase._properties.items()
        if k not in ['Angle', 'RotationAngle']
    }
    _defaults = dict(
        GeometricComponentBase._defaults,
        Method='cylinder',
    )


class FlowerComponent(GeometricComponentBase):
    r"""Flower component."""

    _name = 'Flower'


class FruitComponent(GeometricComponentBase):
    r"""Fruit component."""

    _name = 'Fruit'
    _defaults = dict(
        GeometricComponentBase._defaults,
        Method='sphere',
    )


###############################################
# GENERATOR
###############################################

class PlantGenerator(ParameterCollection):
    r"""Base class for generating plants.

    ClassAttributes:
        _plant_name (str): Name of the plant that this class will
            generate.

    """
    _registry_key = 'crop'
    _plant_name = None
    _name = None
    _help = None
    _default = 'maize'
    _arguments = []
    _properties = dict(
        ParameterCollection._properties,
        data={
            'type': 'string',
            'description': (
                'File containing data that should be used '
                'to set parameters'
            ),
        },
        data_year={
            'type': 'string',
            'description': (
                'Year from which data should be used to set parameters'
            ),
        },
        id={
            'type': 'string',
            'description': ('ID string to be associated with this set of '
                            'property values (e.g. genotype and/or '
                            'genotypic class).'),
        },
        Plastocron={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Time between creation of leaves',
        },
        AgeInc={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Increment that should be used to age the plant',
        },
        AgeMature={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Age after which {class_name} matures',
        },
        AgeSenesce={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Age after which {class_name} senesces',
        },
        AgeRemove={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Age after which {class_name} is removed',
        },
        NMax={
            'type': 'integer',
            'description': 'Maximum photomer number',
        },
    )
    _defaults = dict(
        ParameterCollection._defaults,
        Plastocron=rapidjson.units.Quantity(3.0, 'days'),
        AgeInc=rapidjson.units.Quantity(1.0, 'days'),
        AgeMature=rapidjson.units.Quantity(20.0, 'days'),
        AgeSenesce=rapidjson.units.Quantity(30.0, 'days'),
        AgeRemove=rapidjson.units.Quantity(np.inf, 'days'),
        NMax=20,
    )
    _inherited_properties = [
        'AgeMature', 'AgeSenesce', 'AgeRemove', 'NMax',
    ]
    _components = {
        'Apex': {},
        'Internode': {},
        'Node': {},
    }
    _skip_arguments = {}
    _unit_dimensions = ['length', 'mass', 'time']

    @staticmethod
    def _on_registration(cls):
        if cls._plant_name is not None:
            cls._help = (f'Generate a 3D geometry for a '
                         f'{cls._plant_name} canopy')
            cls._name = cls._plant_name
        cls._properties = copy.deepcopy(cls._properties)
        cls._required = copy.deepcopy(cls._required)
        for component, component_kws in cls._components.items():
            component_cls = get_class_registry().get('plant_component',
                                                     component)
            component_kws = copy.deepcopy(component_kws)
            for k in cls._inherited_properties:
                if k not in component_cls._properties:
                    component_kws.setdefault('properties', {})
                    component_kws['properties'].setdefault(
                        k, copy.deepcopy(cls._properties[k]))
                component_kws.setdefault('external_properties', [])
                component_kws['external_properties'].append(k)
            cls._properties[component] = component_cls.specialize(
                f'{component}_{cls._name}', **component_kws
            )
            if component not in cls._required:
                cls._required.append(component)
        ParameterCollection._on_registration(cls)
        cls._arguments = []
        cls._add_parameter_arguments(cls, cls)

    def __init__(self, param=None, seed=0, verbose=False,
                 verbose_lpy=False, debug=False, debug_param=None,
                 debug_param_prefix=None, unit_system=None,
                 no_class_defaults=False,
                 context=None, **kwargs):
        self.seed = seed
        if debug_param is None:
            debug_param = []
        if debug_param_prefix is None:
            debug_param_prefix = []
        self._debug_param = debug_param
        self._debug_param_prefix = debug_param_prefix
        self._verbose = verbose
        self.verbose_lpy = verbose_lpy
        self._debugging = debug
        self._no_class_defaults = no_class_defaults
        self.unit_system = UnitSet.from_kwargs(
            kwargs, suffix='_units', pop=True
        )
        if param is None:
            param = kwargs
        else:
            if kwargs:
                self.log(f'Invalid keyword arguments '
                         f'{pprint.pformat(kwargs)}')
            assert not kwargs
        param = ParameterDict(param, logger=self)
        self.context = context
        self._cache = ParameterCache()
        super(PlantGenerator, self).__init__(
            '', param, None, required=True,
        )

    @classmethod
    def _property2component(cls, kname):
        for k in cls._components.keys():
            if kname.startswith(k):
                return k
        return None

    @classmethod
    def ids_from_file(cls, fname):
        r"""Determine all of the available ids from the provided file.

        Args:
            fname (str): Data file.

        Returns:
            list: Crop IDs.

        """
        if fname is None:
            return []
        return DataProcessor.from_file(fname).ids

    @classmethod
    def parameters_from_file(cls, args, parameters):
        r"""Calculate parameters based on emperical data.

        Args:
            args (ParsedArguments): Parsed arguments.
            parameters (dict): Parameter dictionary to update.

        Returns:
            dict: Set of parameters calculated from args.

        """
        x = DataProcessor.from_file(args.data)
        for k in x.parameter_names(args.id):
            DictWrapper.remove_prefixed_keys(parameters, f'{k}Func')
            DictWrapper.remove_prefixed_keys(parameters, f'{k}NFunc')
        out = x.parametrize(args.id, args=args, generator=cls)
        parameters.update(out)
        return out

    @property
    def log_prefix_instance(self):
        r"""str: Prefix to use for log messages emitted by this instance."""
        return self._plant_name

    @property
    def crop_class(self):
        r"""str: Crop class."""
        return self.get("id")

    @classmethod
    def add_arguments(cls, parser, **kwargs):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            **kwargs: Additional keyword arguments are passed to parent
                method.

        """
        super(PlantGenerator, cls).add_arguments(parser, **kwargs)
        if not kwargs.get('only_subparser', False):
            cls._add_ids_from_file(parser)

    @classmethod
    def _add_ids_from_file(cls, parser):
        if not cls._name:
            return
        vparser = parser.get_subparser('crop', cls._name)
        ids = DataProcessor.available_ids(cls._name)
        if ids:
            ids_action = vparser.find_argument('id')
            ids_action.choices = (
                ids + ids_action.choices if ids_action.choices
                else ids
            )
            if 'id' in cls._defaults:
                assert cls._defaults['id'] in ids
                ids_action.default = cls._defaults['id']
            else:
                ids_action.default = ids[0]
        years = DataProcessor.available_years(cls._name)
        if years:
            years_action = vparser.find_argument('data_year')
            years_action.choices = (
                years + years_action.choices if years_action.choices
                else years
            )
            if 'data_year' in cls._defaults:
                assert cls._defaults['data_year'] in years
                years_action.default = cls._defaults['data_year']
            else:
                years_action.default = years[0]

        available = set()
        for action in vparser._actions:
            available |= set(action.option_strings + [action.dest])

    @property
    def lsystem(self):
        r"""str: Lsystem for the generator."""

        try:

            def header(msg):
                return '\n'.join([
                    '',
                    60 * '#',
                    f'## {msg}',
                    60 * '#',
                    '',
                ])

            args_apex = self.parameters['Apex'].lsystem_args()
            nargs_apex = len(args_apex.split(','))
            args_apex_init = ','.join(nargs_apex * ['0'])
            out = [
                header("External parameters"),
                'extern(RUNTIME_PARAM = {})',
                'extern(OUTPUT_TIME = 27)',
                'extern(DERIVATION_LENGTH = OUTPUT_TIME)'
                '',
                header('Python functions'),
                'from canopy_factory.utils import get_class_registry',
                f'generator_cls = get_class_registry().get('
                f'"crop", "{self._name}")',
                'generator = generator_cls('
                'context=context(), **RUNTIME_PARAM)',
                'NMax = generator.NMax()',
                'PLASTOCHRON = generator.Plastocron()',
                'StartingAngle = generator.sample_dist(\"uniform\", 0, 360)',
                'if generator.verbose_lpy:',
                '    print(f"OUTPUT_TIME = {OUTPUT_TIME}")',
                '    print(f"DERIVATION_LENGTH = {DERIVATION_LENGTH}")',
                '    print(f"PLASTOCHRON = {PLASTOCHRON}")',
                '',
                header('Begin LSystem syntax'),
                'module A',
            ]
            out += [f'module {k}' for k in self._components.keys()]
            out += [
                '',
                f'Axiom: @Gc/(StartingAngle)Apex({args_apex_init})@Ge',
                'derivation length: DERIVATION_LENGTH',
                '',
                'production:',
                '',
            ]
            for k in self._components.keys():
                if k not in self.parameters:
                    continue
                v = self.parameters[k]
                args = v.lsystem_args(for_rule=True)
                vrule = ['    ' + x for x in v.production]
                if not vrule:
                    continue
                out += [
                    f'{v.name}({args}):',
                ] + vrule + ['']
            out += [
                'homomorphism:',
                'maximum depth: 3',
                '',
            ]
            for k in self._components.keys():
                if k not in self.parameters:
                    continue
                v = self.parameters[k]
                args = v.lsystem_args(for_rule=True)
                vrule = ['    ' + x for x in v.homomorphism]
                if not vrule:
                    continue
                out += [f'{v.name}({args}):'] + vrule + ['']
            out += [
                'interpretation:',
                '',
                'endlsystem',
                '',
            ]
        except AttributeError:
            import traceback
            print(traceback.format_exc())
            raise
        return '\n'.join(out)
