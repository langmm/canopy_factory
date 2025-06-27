import os
import pprint
import pdb
import copy
import functools
import numpy as np
import pandas as pd
import scipy
from datetime import datetime
from openalea.lpy import Lsystem
import openalea.plantgl.all as pgl
import openalea.plantgl.math as pglmath
from openalea.plantgl.math import Vector2, Vector3, Vector4
from openalea.plantgl.scenegraph import (
    NurbsCurve2D, NurbsCurve, NurbsPatch)
# from openalea.plantgl.all import Tesselator
from yggdrasil import rapidjson, units
from canopy_factory import utils
from canopy_factory.utils import (
    parse_units, parse_quantity, parse_solar_time, parse_axis, parse_color,
    RegisteredMetaClass, RegisteredClassBase, get_class_registry,
    NoDefault, scale_factor,
    cached_property, readonly_cached_property,
)
from canopy_factory.cli import TaskBase


############################################################
# LPy parametrization class
#  - 'age' indicates the time since germination
#  - 'n' indicates the phytomer count
############################################################


class PlantParameterBase(RegisteredClassBase):

    _registry_key = 'plant_parameter'


def DelayedPlantParameter(name):

    def parameter_class():
        return get_class_registry().get('plant_parameter', name)

    class DelayedPlantParameterMeta(type):

        def __call__(self, *args, **kwargs):
            return parameter_class()(*args, **kwargs)

        def __getattr__(self, k):
            return getattr(parameter_class(), k)

    class _DelayedPlantParameter(object, metaclass=DelayedPlantParameterMeta):
        pass

    return _DelayedPlantParameter


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
            that debug mode should be turned on for.
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
        _dependencies (list): Root level properties that this parameter
            uses.
        _prefix (str): Prefix that should be used for child parameters
            used by this parameter.
        _subschema_keys (list): JSON schema properties that contain
            schemas.
        _component_properties (list): Properties that can be defined for
            all parameters associated with a component.
        _attribute_properties (list): Properties that should be
            accessible directly as class attributes.
        _index_var (list): Variables that will be supplied as input to
            calls to generate parameters.
        _components (list): Plant components that may be used with
            _component_properties.

    """

    _name = 'simple'
    _properties = {
        '': {
            'type': ['string', 'null', 'number', 'boolean']
        },
    }
    _defaults = {}
    _aliases = {}
    _required = []
    _variables = []
    _dependencies = []
    _prefix = None
    _subschema_keys = ['oneOf', 'allOf']
    _component_properties = []
    _attribute_properties = []
    _index_var = ['X', 'N', 'Age']
    _components = []

    def __init__(self, name='', param=None, parent=None, defaults=None,
                 seed=0, verbose=False, debug=False, debug_param=None,
                 required=False, length_units=None, time_units=None,
                 **kwargs):
        super(SimplePlantParameter, self).__init__()
        self._key_stack = []
        self._cache = {}
        self._generators = {}
        self.name = name
        self.parameters = {}
        self.defaults = {}
        self.parent = parent
        self.seed = seed
        if debug_param is None:
            debug_param = []
        self._debug_param = debug_param
        self._verbose = verbose
        self._debugging = debug
        self.required = required
        self.length_units = length_units
        self.time_units = time_units
        self.initialized = False
        self.child_parameters = self.parameter_names(self.fullname, 'children')
        self.core_paremeters = self.parameter_names(self.fullname, 'core')
        self.valid_parameters = self.child_parameters + self.core_paremeters
        if defaults is None:
            defaults = {}
        defaults = self.select_local(defaults, ensure_copy=True)
        self.defaults = dict(self._defaults, **defaults)
        self.log(f'Defaults = {pprint.pformat(self.defaults)}')
        if param is None:
            assert parent is None
            param = kwargs
        else:
            if kwargs:
                self.log(f'Invalid keyword arguments '
                         f'{pprint.pformat(kwargs)}')
            assert not kwargs
        self.update(param, in_init=True)

    def clear(self):
        r"""Clear the parameter class."""
        self.initialized = False
        self.parameters.clear()
        self.defaults.clear()

    def update(self, param, in_init=False):
        r"""Update the parameters.

        Args:
            param (dict): New parameters.
            in_init (bool, optional): If True, this is the first call
                to create the parameter.

        """
        param0 = copy.deepcopy(param)
        for k, v in self._aliases.items():
            if k in param:
                param[v] = param.pop(k, None)
        self.log(f'Parsing param:\n{pprint.pformat(param)}')
        schema = self.schema()
        self.initialized = self._parse_single(param, schema)
        if self.initialized:
            self.log(f'Parsed param:\n{pprint.pformat(self.parameters)}')
            for k in self.parameters.keys():
                param.pop(f'{self.fullname}{k}', None)
            if self.parent is None:
                for k in self.all_component_properties:
                    param.pop(k, None)
            self.log(f'BEFORE:\n{pprint.pformat(param0)}')
            self.log(f'SCHEMA:\n{pprint.pformat(schema)}')
            self.log(f'AFTER:\n{pprint.pformat(param)}')
        elif self.required:
            self.error(rapidjson.NormalizationError,
                       'Failed to initialize required parameter')
        else:
            self.log('MISSING PARAMETERS')
        if self.parent is None and param:
            self.debugging = True
            self.error(AssertionError,
                       f'Unparsed param:\n{pprint.pformat(param)}')
        if self.initialized:
            msg = f'INITIALIZED:\n{pprint.pformat(self.contents)}'
            debug = ((not in_init) or self.parent == self.root)
            self.log(msg, debug=debug)

    def update_defaults(self, param):
        r"""Update defaults based on parameters.

        Args:
            param (dict): Parameters to update defaults from.

        """
        for k in self._component_properties:
            kcomp = f'{self.component}{k}'
            v = param.get(kcomp, self.defaults.get(kcomp, NoDefault))
            if v is NoDefault:
                continue
            self.defaults.setdefault(k, v)

    def select_local_for_init(self, param):
        r"""Select parameters from initial set.

        Args:
            param (dict): Parameters to select from.

        Returns:
            dict: Selected parameters with the current prefix removed.

        """
        self.update_defaults(param)
        self.log(f'DEFAULTS = {pprint.pformat(self.defaults)}')
        return self.select_local(param, include_existing=True,
                                 include_defaults=True,
                                 use_fullname=True)

    def select_local(self, param, ensure_copy=False,
                     include_existing=False, include_defaults=False,
                     use_fullname=False, schema=None):
        r"""Select parameters from a dictionary that match the current
        prefix and return them with the prefix removed.

        Args:
            param (dict): Parameters to select from.
            ensure_copy (bool, optional): If True, the returned dictionary
                will be a copy.
            include_existing (bool, optional): If True, existing
                parameters should be moved into the output dictionary.
            include_defaults (bool, optional): If True, default values
                should be included in the output dictionary.
            use_fullname (bool, optional): If True, param contains fully
                specified parameter names and the full name should be
                used as the prefix.
            schema (dict, optional): JSON schema that should be used to
                only select relevant properties.

        Returns:
            dict: Selected parameters with the current prefix removed.

        """
        param = {
            self.remove_prefix(k, use_fullname=use_fullname): v
            for k, v in param.items()
            if self.prefixes(k, use_fullname=use_fullname)
        }
        if ensure_copy:
            param = copy.deepcopy(param)
        if include_existing:
            param = dict(self.parameters, **param)
            self.parameters = {}
        if include_defaults:
            for k, v in self.defaults.items():
                param.setdefault(k, v)
        if schema is not None:
            param = {
                k: v for k, v in param.items()
                if self.schema_contains(schema, k, value=v)
            }
        return param

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
        # for k, v in schema.get('properties', {}).items():
        #     if not isinstance(v, type):
        #         continue
        #     # TODO
        return False

    @classmethod
    def specialize(cls, name, properties=None, defaults=None, exclude=[],
                   **kwargs):
        r"""Created a specialized version of this class.

        Args:
            name (str): Name of the root property for the class.
            properties (dict, optional): Properties that should be added.
            defaults (dict, optional): Defaults that should be added.
            exclude (list, optional): Properties that should be excluded.
            **kwargs: Additional keyword arguments are added as attributes
                to the resulting class.

        """
        class_dict = copy.deepcopy(kwargs)
        class_dict.setdefault('_name', name)
        class_dict.setdefault(
            '_properties', {
                k: copy.deepcopy(v) for k, v in cls._properties.items()
                if k not in exclude
            })
        class_dict.setdefault(
            '_required', [
                k for k in cls._required if k not in exclude
            ])
        if properties:
            class_dict['_properties'].update(**properties)
        if defaults:
            class_dict.setdefault(
                '_defaults', {
                    k: copy.deepcopy(v) for k, v in cls._defaults.items()
                    if k not in exclude
                })
            class_dict['_defaults'].update(**defaults)
        created = RegisteredMetaClass(f'Created{name}', (cls,), class_dict)
        return created

    @property
    def prefix(self):
        r"""str: Prefix that child parameters will have."""
        out = self.name
        if self._prefix and not out.endswith(self._prefix):
            out += self._prefix
        return out

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
        r"""str: Component that this property belongs to."""
        key = self.fullname
        if not key:
            return None
        for i in range(1, len(key)):
            if key[i].isupper():
                return key[:i]
        raise RuntimeError(f'Could not extract component from \"{key}\"')

    @property
    def root(self):
        r"""PlantParameterBase: Root parameter."""
        if self.parent is not None:
            return self.parent.root
        return self

    @property
    def verbose(self):
        r"""bool: If True, log messages will be emitted."""
        if self.debugging:
            return True
        if self.parent is not None:
            return self.parent.verbose
        return self._verbose

    @property
    def debug_param(self):
        r"""list: Parameters that debug mode should be enabled for."""
        if self.parent is not None:
            return self.parent.debug_param
        return self._debug_param

    @property
    def debugging(self):
        r"""bool: True if debugging is active."""
        if self.fullname.startswith(tuple(self.debug_param)):
            return True
        if self.parent is not None:
            return self.parent.debugging
        return self._debugging

    @debugging.setter
    def debugging(self, value):
        if self.parent:
            self.parent.debugging = value
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
        if k in self._attribute_properties:
            return self.parameters.get(k, None)
        if k not in self.valid_parameters:
            self.error(AttributeError, f'{self}: {k}')
        return functools.partial(self.getfull, k, k0=k)

    def getfull(self, k, age, n, x=None, **kwargs):
        r"""Get a parameter value with the index fully specified as
        arguments.

        Args:
            k (str): Name of parameter to return without any prefixes.
            age (float): Age of the component that the parameter should
                be returned for.
            n (int):  Phytomer count of the component that the parameter
                should be returned for.
            x (float, optional): Position along the component that the
                parameter should be returned for.
            **kwargs: Additional keyword arguments are passed to get.

        Returns:
            object: Parameter value.

        """
        idx = (age, n, x)
        return self.get(k, idx=idx, **kwargs)

    def set(self, k, value, k0=None):
        r"""Set a parameter value.

        Args:
            k (str): Name of parameter to set, without any prefixes.
            value (object): Value to set parameter to.
            k0 (str): Full name of the parameter including any parent
                prefixes. If not provided, the full name will be
                created based on the current prefix.

        """
        raise NotImplementedError

    @property
    def children(self):
        r"""iterable: Child parameter instances."""
        for v in self.parameters.values():
            if isinstance(v, PlantParameterBase) and v.initialized:
                yield v

    @property
    def dependencies(self):
        r"""list: Set of variables that this parameter is dependent on."""
        out = copy.deepcopy(self._dependencies)
        out += [self.parameters.get(k, None) for k in self._variables
                if k in self.parameters]
        for child in self.children:
            out += child.dependencies
        return list(set(out))

    @property
    def component_properties(self):
        r"""list: Set of component-wide properties for this and
        child parameters without component prefixes."""
        out = copy.deepcopy(self._component_properties)
        for child in self.children:
            out += child.component_properties
        return out

    @property
    def all_component_properties(self):
        r"""list: Set of component-wide properties for this and
        child parameters with component prefixes."""
        properties = self.component_properties
        out = []
        for comp in self._components:
            out += [f'{comp}{k}' for k in properties]
        return out

    @property
    def constant_var(self):
        r"""list: Set of index variables that this parameter is
        independent of."""
        dependencies = self.dependencies
        return [k for k in self._index_var if k not in dependencies]

    def _nullify_constant_var(self, idx, var=None, context=None):
        if var is None:
            var = self.constant_var
        if isinstance(var, list):
            for v in var:
                idx = self._nullify_constant_var(idx, var=v,
                                                 context=context)
            return idx
        elif var is None:
            return idx
        age, n, x = idx[:]
        if var == 'Age':
            age = None
        elif var == 'X':
            x = None
        elif var == 'N':
            n = None
        else:
            raise ValueError(f"Unsupported constant var \"{var}\"")
        return (age, n, x)

    def get(self, k, default=NoDefault, idx=None, cache_key=None,
            k0=None, return_other=None, **kwargs):
        r"""Get a parameter value.

        Args:
            k (str): Name of parameter to return, without any prefixes.
            default (object, optional): Default to return if the
                parameter is not set.
            idx (tuple, optional): Simulation index to get the parameter
                value for. If not provided, the current index will be
                used based on the last parameter accessed with an index.
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
        kwargs['return_other'] = return_other
        if idx is None:
            idx = self._current_idx(cache_key=cache_key)
        kidx = self._nullify_constant_var(idx, context=k)
        age, n, x = kidx[:]
        if k0 is None:
            is_child = True
            k0 = f'{self.fullname}{k}'
        else:
            is_child = self.prefixes(k)
            if is_child:
                k = self.remove_prefix(k)
        if k == 'Age':
            assert age is not None
            return age
        elif k == 'N':
            assert n is not None
            return n
        elif k == 'X':
            assert x is not None
            return x
        elif k0 not in self.valid_parameters:
            if is_child:
                if default is not NoDefault:
                    return default
                self.error(KeyError,
                           f'Missing child parameter \"{k}\" '
                           f'(k0=\"{k0}\")?', debug=True)
            self.log(f'External parameter \"{k0}\"')
            return self.root.get(k0, default=default, idx=idx, **kwargs)
        if k and k not in self.parameters:
            for kk, vv in self.parameters.items():
                if isinstance(vv, PlantParameterBase) and vv.prefixes(k):
                    return vv.get(k, idx=kidx, k0=k0, cache_key=cache_key,
                                  default=default, **kwargs)
            else:
                self.error(KeyError, f"Unsupported var \"{k0}\"")
        if not return_other:
            out = self._push_key(k0, *kidx, cache_key=cache_key)
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
                elif return_other == 'parameters':
                    v = v.parameters
                elif return_other == 'instance':
                    pass
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

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        return self.parameters['']

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
            if name not in out:
                out.insert(0, name)
        if scope in ['all', 'children']:
            for k, v in cls._properties.items():
                if isinstance(v, type):
                    if v._prefix is None or name.endswith(v._prefix):
                        kname = f'{name}{k}'
                    else:
                        kname = f'{name}{k}{v._prefix}'
                    out += v.parameter_names(kname)
        return out

    def _extract_parameters(self, schema):
        parameters = {'properties': {}, 'required': []}
        for k in list(schema.get('properties', {}).keys()):
            if isinstance(schema['properties'][k], type):
                parameters['properties'][k] = schema['properties'].pop(k)
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
                      parameters=None, required=None, idstr=''):
        if idstr:
            self.log(f'{idstr}: {schema}')
        if child_param is None:
            child_param = self.select_local_for_init(param)
            self.log(f'CHILDREN:\n{pprint.pformat(child_param)}')
        if required is None:
            required = self.required
        # Parse subschemas w/ parameter classes via recursion
        try:
            for k in self._subschema_keys:
                if k not in schema:
                    continue
                krequired = (k == 'allOf')
                kparam = [
                    self._extract_parameters(x) for x in schema[k]
                ]
                if not any(kparam):
                    continue
                results = [
                    self._parse_single(
                        param, x, child_param=child_param,
                        idstr=f' {k}[{i}]', required=krequired,
                        parameters=kparam[i]
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
                        raise rapidjson.NormalizationError('No matches')
                elif k == 'allOf':
                    if sum(results) != len(results):
                        invalid = [
                            schema[k][i] for i, v in
                            enumerate(results) if not v
                        ]
                        raise rapidjson.NormalizationError(
                            f'Not all match:\n'
                            f'{pprint.pformat(invalid)}\n'
                            f'{pprint.pformat(child_param)}'
                        )
                else:
                    self.error(NotImplementedError,
                               f'Unsupported schema key {k}')
                schema.pop(k)
            # Split schema & param into regular parameters and parameter
            #   classes
            if parameters is None:
                parameters = self._extract_parameters(schema)
            child_param_schema = {
                k: v for k, v in child_param.items()
                if self.schema_contains(schema, k, value=v)
            }
            # Parse regular schema parameters
            if schema:
                try:
                    out = rapidjson.normalize(child_param_schema, schema)
                except rapidjson.NormalizationError as e:
                    raise rapidjson.NormalizationError(
                        f'Normalization error: {e.args[0]}\n'
                        f'Schema param:\n'
                        f'{pprint.pformat(child_param_schema)}'
                    )
            # Parse parameter classes
            missing = []
            for k, v in parameters.get('properties', {}).items():
                self.log(f'Adding{idstr} child parameter \"{k}\": {v}')
                krequired = (k in parameters.get('required', []))
                if isinstance(child_param.get(k, None),
                              PlantParameterBase):
                    out[k] = child_param.pop(k)
                    out[k].update(param)
                else:
                    out[k] = v(
                        k, param, defaults=self.defaults,
                        parent=self, required=(required and krequired),
                    )
                if (not out[k].initialized) and krequired:
                    missing.append(k)
            if missing:
                raise rapidjson.NormalizationError(
                    f'Failed to initialized child parameter '
                    f'instance(s): {missing}'
                )
            # Update parameters on the class
            self.parameters.update(**out)
            return True
        except rapidjson.NormalizationError as e:
            if required:
                self.error(rapidjson.NormalizationError, e.args[0])
            self.log(f'Ignored error: {e.args[0]}')
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
        prefix = ''
        key = self._current_key
        if key is not None:
            k, age, n, x = key[:]
            if x is None:
                x_str = ''
            else:
                x_str = f',x={x}'
            prefix = f'{k}[{age},{n}{x_str}]: '
        return prefix

    def log(self, message='', force=False, debug=False):
        r"""Emit a log message.

        Args:
            message (str, optional): Log message.
            force (bool, optional): If True, print the log message even
                if self.verbose is False.
            debug (bool, optional): If True, set a debug break point if
                debugging enabled.

        """
        return super(SimplePlantParameter, self).log(
            message=message, force=force,
            debug=(debug and self.debugging),
            prefix=f'{self.log_prefix_instance}: {self.log_prefix_stack}'
        )

    def error(self, error_cls, message='', debug=False):
        r"""Raise an error, adding context to the message.

        Args:
            error_cls (type): Error class.
            message (str, optional): Error message.
            debug (bool, optional): If True, set a debug break point.

        """
        return super(SimplePlantParameter, self).error(
            error_cls, message=message,
            debug=(debug or self.debugging),
            prefix=f'{self.log_prefix_instance}: {self.log_prefix_stack}'
        )

    # Methods for tracking things
    def _current_idx(self, cache_key=None):
        current_key = cache_key
        if current_key is None:
            current_key = self._current_key
        if current_key is None:
            current_key = (None, None, None, None)
        idx = current_key[1:]
        return idx

    def _push_key(self, param, age, n, x, cache_key=None):
        if self.parent:
            return self.parent._push_key(param, age, n, x,
                                         cache_key=cache_key)
        if cache_key is None:
            key = (param, age, n, x)
        else:
            key = cache_key
        if key in self._cache:
            # self.log(f"Using cached ({key})")
            return self._cache[key]
        self._key_stack.append(key)

    def _pop_key(self, out=None):
        if self.parent:
            return self.parent._pop_key(out=out)
        if out is not None:
            self._cache[self._current_key] = out
        self.log(str(out))
        self._key_stack.pop()

    @property
    def _current_key(self):
        if self.parent:
            return self.parent._current_key
        if self._key_stack:
            return self._key_stack[-1]
        return None

    # @property
    # def current_component(self):
    #     r"""str: Name of component currently being generated."""
    #     if self.parent:
    #         return self.parent.current_component
    #     key = self._current_key
    #     if key is None:
    #         return None
    #     for i in range(1, len(key[0])):
    #         if key[0][i].isupper():
    #             return key[0][:i]
    #     raise RuntimeError(f'Could not extract component from \"{key[0]}\"')

    @property
    def generator(self):
        r"""np.random.Generator: Current random number generator."""
        if self.parent:
            return self.parent.generator
        seed = self.seed
        if self._current_key and (self._current_key[2] is not None):
            seed += self._current_key[2]
        if seed not in self._generators:
            self.log(f"Creating generator for seed = {seed}")
            self._generators[seed] = np.random.default_rng(
                seed=seed)
        return self._generators[seed]


class OptionPlantParameter(SimplePlantParameter):
    r"""Class for parameter with exclusive options."""

    _name = None
    _name_option = None
    _properties = {}
    _options = {}
    _default = NoDefault

    @staticmethod
    def _on_registration(cls):
        opt = cls._name_option
        if opt is None:
            opt = cls._name
        SimplePlantParameter._on_registration(cls)
        if opt is not None:
            cls._properties = copy.deepcopy(cls._properties)
            cls._properties[opt] = {
                'enum': list(sorted(cls._options.keys()))
            }
            if opt not in cls._required:
                cls._required = copy.deepcopy(cls._required)
                cls._required.insert(0, opt)
        if cls._default is not NoDefault:
            cls._defaults = copy.deepcopy(cls._defaults)
            cls._defaults[opt] = copy.deepcopy(cls._default)

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        opt = cls._name_option
        if opt is None:
            opt = cls._name
        out = super(OptionPlantParameter, cls).schema()
        out['oneOf'] = []
        for k, v in cls._options.items():
            kprop = {}
            krequired = out.get('required', []) + v
            assert opt in krequired
            for kreq in krequired:
                kprop[kreq] = copy.deepcopy(out['properties'][kreq])
            kprop[opt]['enum'] = [k]
            if cls._default is NoDefault:
                this_param = set(v)
                other_param = []
                for kk, vv in cls._options.items():
                    if kk != k:
                        other_param += vv
                unique_param = this_param - set(other_param)
                if unique_param:
                    kprop[opt]['default'] = k
            out['oneOf'].append({
                'properties': kprop,
                'required': krequired,
            })
        out.pop('required', None)
        return out


class FunctionPlantParameter(OptionPlantParameter):
    r"""Class for a function."""

    _name = 'Func'
    _properties = {
        'FuncVar': {'type': 'string'},
        'FuncVarNorm': {'type': 'string'},
        'FuncVarMax': {'type': 'string'},
        'Slope': {'type': 'number'},
        'Intercept': {'type': 'number'},
        'Amplitude': {'type': 'number', 'default': 1.0},
        'Period': {'type': 'number', 'default': 2.0 * np.pi},
        'XOffset': {'type': 'number', 'default': 0.0},
        'YOffset': {'type': 'number', 'default': 0.0},
        'Exp': {'type': 'number'},
        'XVals': {
            'oneOf': [
                {'type': 'ndarray', 'subtype': 'float'},
                {'type': 'array', 'items': {'type': 'number'},
                 'minItems': 2, 'maxItems': 2},
            ],
        },
        'YVals': {'type': 'ndarray', 'subtype': 'float'},
        'Curve': DelayedPlantParameter('curve'),
        'CurvePatch': DelayedPlantParameter('curve_patch'),
        'Method': {'type': 'string'},
        'Function': {'type': 'function'},
    }
    _options = {
        'linear': ['Slope', 'Intercept'],
        'sin': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'cos': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'tan': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'pow': ['Amplitude', 'Exp', 'XOffset', 'YOffset'],
        'interp': ['XVals', 'YVals'],
        'curve': ['Curve'],
        'curve_patch': ['CurvePatch'],
        'method': ['Method'],
        'user': ['Function'],
    }
    _required = ['FuncVar']
    _variables = ['FuncVar']
    prefix = ''

    @property
    def xvar(self):
        r"""str: Variable that this function parameter takes as input."""
        return self.parameters['FuncVar']

    @property
    def normvar(self):
        r"""str: Variable that should be used to normalize the input."""
        return self.parameters.get('FuncVarNorm', None)

    @property
    def maxvar(self):
        r"""str: Variable that contains the maximum value under which
        the function applies."""
        return self.parameters.get('FuncVarMax', None)

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        kwargs['idx'] = idx
        func = self.parameters['Func']
        v = self.root.get(self.xvar, **kwargs)
        if self.maxvar is not None:
            vmax = self.get(self.maxvar, **kwargs)
            if v > vmax:
                return 1.0
        if self.normvar is not None:
            v /= float(self.get(self.normvar, **kwargs))
        if callable(func):
            return func(v)
        elif func == 'linear':
            slope = self.parameters['Slope']
            intercept = self.parameters['Intercept']
            return slope * v + intercept
        elif func in ['sin', 'cos', 'tan']:
            A = self.parameters['Amplitude']
            period = self.parameters['Period']
            xoffset = self.parameters['XOffset']
            yoffset = self.parameters['YOffset']
            ftrig = getattr(np, func)
            return (
                (A * ftrig(2.0 * np.pi * (v + xoffset) / period))
                + yoffset)
        elif func == 'pow':
            A = self.parameters['Amplitude']
            exp = self.parameters['Exp']
            xoffset = self.parameters['XOffset']
            yoffset = self.parameters['YOffset']
            return (A * pow(v - xoffset, exp)) + yoffset
        elif func == 'interp':
            xvals = self.parameters['XVals']
            yvals = self.parameters['YVals']
            if isinstance(xvals, (tuple, list)):
                xvals = np.linspace(xvals[0], xvals[1], len(yvals))
            f = scipy.interpolate.interp1d(xvals, yvals)
            return f(v)
        elif func == 'curve':
            curve = self.get('Curve', **kwargs)
            return CurvePlantParameter.sample_curve(curve, v)
        elif func == 'curve_patch':
            patch = self.get('CurvePatch', **kwargs)
            return CurvePatchPlantParameter.sample_curve_patch(patch, v)
        elif func == 'method':
            method = self.get('Method', **kwargs)
            assert self.root.hasmethod(method)
            return getattr(self.root, method)(v)
        elif func == 'user':
            function = self.get('Function', **kwargs)
            return function(v)
        else:
            raise ValueError(f"Unsupported function name \"{func}\"")

    @classmethod
    def specialize(cls, var, normvar=None, maxvar=None, exclude=[],
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
            exclude (list, optional): Properties that should be excluded.
            **kwargs: Additional keyword arguments are passed to the
                base class's method.

        """
        exclude = exclude + [
            'FuncVar', 'FuncVarNorm', 'FuncVarMax'
        ]
        kwargs.setdefault('xvar', var)
        kwargs.setdefault('normvar', normvar)
        kwargs.setdefault('maxvar', maxvar)
        var = kwargs['xvar']
        normvar = kwargs['normvar']
        maxvar = kwargs['maxvar']
        kwargs.setdefault('prefix', var)
        name = f'{var}Func'
        kwargs.setdefault('_name_option', 'Func')
        out = super(FunctionPlantParameter, cls).specialize(
            name, exclude=exclude, **kwargs
        )
        if var not in out._dependencies:
            out._dependencies = copy.deepcopy(out._dependencies)
            out._dependencies.append(var)
        for x in [normvar, maxvar]:
            if x is None:
                continue
            if x not in out._required:
                out._required.append(x)
            if x not in out._properties:
                out._properties[x] = {'type': 'number'}
        return out


class DistributionPlantParameter(OptionPlantParameter):
    r"""Class for a distribution."""

    _name = 'Dist'
    _default = 'normal'
    _properties = {
        'Mean': {'type': 'number', 'default': 1.0},
        'StdDev': {'type': 'number', 'default': 0.2},
        'Bounds': {'type': 'array', 'items': {'type': 'number'},
                   'minItems': 2, 'maxItems': 2, 'default': [0.0, 1.0]},
    }
    _options = {
        'normal': ['Mean', 'StdDev'],
        'uniform': ['Bounds'],
    }
    prefix = ''

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        profile = self.parameters['Dist']
        if profile == 'normal':
            return self.generator.normal(
                self.parameters['Mean'],
                self.parameters['StdDev'],
            )
        elif profile == 'uniform':
            return self.generator.uniform(
                *self.parameters['Bounds'],
            )
        else:
            raise ValueError(f"Unsupported variance profile "
                             f"\"{profile}\"")


class ScalarPlantParameter(SimplePlantParameter):
    r"""Class for scalar parameters that will have a spread by default
    and can have a dependence on age, n, or x."""

    _name = 'scalar'
    _properties = {
        '': {'type': 'number'},
        'Dist': DistributionPlantParameter,
        'Func': FunctionPlantParameter,
        'XFunc': FunctionPlantParameter.specialize('X'),
        'NFunc': FunctionPlantParameter.specialize(
            'N', normvar='Max',
        ),
        'AgeFunc': FunctionPlantParameter.specialize(
            'Age', normvar='Mature', maxvar='Mature',
            properties={'Max': {'type': 'number'}},
            defaults={'Slope': 1.0, 'Intercept': 0.0},
            _default='linear',
        ),
    }
    _component_properties = ['NMax', 'AgeMax', 'AgeMature']

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        out = self.parameters.get('', 1.0)
        self.log(f'base = {out}')
        for k in ['Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc']:
            v = self.parameters.get(k, None)
            if v and v.initialized:
                ifactor = self.parameters[k].generate(idx, **kwargs)
                self.log(f'{k} = {ifactor}')
                out *= ifactor
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
        },
        'Symmetry': {
            'type': 'array', 'items': {'type': 'integer'},
        },
        'Closed': {'type': 'boolean', 'default': False},
        'Reverse': {'type': 'boolean', 'default': False},
        'Thickness': {'type': 'number'},
        'Patch': DelayedPlantParameter('curve_patch'),
        'PatchVar': {'type': 'string'},
        'PatchNorm': {'type': 'number'},
        'PatchMin': {'type': 'number'},
        'PatchMax': {'type': 'number'},
    }
    _required = []
    _prefix = 'Curve'
    _required_curve = ['ControlPoints']
    _required_patch = ['Patch', 'PatchVar']
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
            ] + ['PatchVar']

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

    def update_defaults(self, param):
        r"""Update defaults based on parameters.

        Args:
            param (dict): Parameters to update defaults from.

        """
        def _get(k):
            return param.get(f'{self.fullname}{k}',
                             self.defaults.get(k, NoDefault))

        if self._allow_patch and not all((_get(k) is not NoDefault)
                                         for k in self._required_curve):
            for k in self._shared:
                v = _get(k)
                if v is NoDefault:
                    continue
                for kchild in ['Start', 'End']:
                    self.defaults.setdefault(f'Patch{kchild}Curve{k}', v)
        super(CurvePlantParameter, self).update_defaults(param)

    def generate(self, idx, return_points=False, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            return_points (bool, optional): If True, the control points
                for the curve will be returned instead of the curve.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        kwargs['idx'] = idx
        if ((self._allow_patch and 'Patch' in self.parameters
             and self.parameters['Patch'].initialized)):
            if return_points:
                kwargs.pop('idx')
                patch = self.parameters['Patch']
                return (
                    patch.parameters('StartCurve').generate(
                        idx, return_points=return_points, **kwargs),
                    patch.parameters('EndCurve').generate(
                        idx, return_points=return_points, **kwargs),
                )
            patch = self.get('Patch', **kwargs)
            tvar = self.get('PatchVar', **kwargs)
            tnorm = self.parameters.get('PatchNorm', None)
            tmin = self.parameters.get('PatchMin', None)
            tmax = self.parameters.get('PatchMax', None)
            t = self.get(tvar, **kwargs)
            curve = CurvePatchPlantParameter.sample_curve_patch(
                patch, t, tnorm=tnorm, tmin=tmin, tmax=tmax)
        else:
            points = self.get('ControlPoints', **kwargs)
            symmetry = self.parameters.get('Symmetry', None)
            closed = self.parameters.get('Closed', False)
            reverse = self.parameters.get('Reverse', False)
            thickness = self.parameters.get('Thickness', None)
            curve = self.create_curve(
                points, symmetry=symmetry, closed=closed,
                reverse=reverse, thickness=thickness,
                return_points=return_points,
                # factor=factor,
            )
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
    def create_curve(cls, points, knots=None, uniform=False,
                     stride=60, degree=3, symmetry=None, closed=False,
                     reverse=False, thickness=None, factor=None,
                     return_points=False):
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
        if thickness is not None:
            closed = True
            if points.shape[1] != 2:
                raise NotImplementedError('Thickness for 3D curve')
            nthickness1 = int(np.ceil(points.shape[0] / 2))
            nthickness2 = int(np.floor(points.shape[0] / 2))
            assert (nthickness1 + nthickness2) == points.shape[0]
            thickness1 = np.linspace(0, thickness, nthickness1)
            thickness2 = thickness1[:nthickness2]
            thickness0 = np.concatenate([thickness1, thickness2[::-1]])
            points_bottom = points.copy()
            points_bottom[:, 1] -= thickness0
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
    _properties = {
        'Start': CurvePlantParameter.specialize(
            'start_curve', exclude=CurvePlantParameter._patch_properties),
        'End': CurvePlantParameter.specialize(
            'end_curve', exclude=CurvePlantParameter._patch_properties),
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
            t /= tnorm
            if tmin is not None:
                tmin /= tnorm
            if tmax is not None:
                tmax /= tnorm
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

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        kwargs['idx'] = idx
        start = self.get('Start', **kwargs)
        end = self.get('End', **kwargs)
        patch = self.create_curve_patch([start, end])
        return patch


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


class PlantGenerator(ParameterCollection):
    r"""Base class for generating plants.

    ClassAttributes:
        _plant_name (str): Name of the plant that this class will
            generate.

    """
    _plant_name = None
    _parameters = {
        'scalar': [
            'LeafLength', 'LeafWidth',
            'BranchAngle', 'RotationAngle',
            'InternodeLength', 'InternodeWidth',
        ]
    }
    _components = [
        'Leaf', 'Internode', 'Branch',
    ]

    @property
    def log_prefix_instance(self):
        r"""str: Prefix to use for log messages emitted by this instance."""
        return self._plant_name

    def sample_dist(self, profile, *args, **kwargs):
        r"""Sample a distribution using the current random number
        generator.

        Args:
            profile (str): Name of the profile that should be sampled.
            *args, **kwargs: Additional arguments are passed to the
                generator method for the specified profile.

        """
        return getattr(self.generator, profile)(*args, **kwargs)


class MaizeGenerator(PlantGenerator):
    r"""Class for generating maize plant geometries."""

    _plant_name = 'maize'
    _properties = dict(
        PlantGenerator._properties,
        **{
            'leaf_data_file': {'type': 'string'},
            'leaf_data_units': {'type': 'string'},
            'leaf_data_time': {'type': 'number', 'default': 27},
            'crop_class': {'type': 'string', 'default': 'WT'},
            'unfurl_leaves': {'type': 'boolean', 'default': False},
        }
    )
    _parameters = {
        'scalar': [
            'LeafBend', 'LeafTwist',
        ],
        'curve': [
            'LeafProfile',  # 'LeafBend',
        ],
    }
    _aliases = dict(
        PlantGenerator._aliases,
        LeafThickness='LeafProfileCurveThickness',
        LeafLengthUnfurled='LeafProfileCurvePatchMax',
    )
    _defaults = dict(
        PlantGenerator._defaults,
        LeafThickness=0.1,  # relative to leaf width
        LeafLengthUnfurled=0.3,   # relative to leaf length
        LeafProfileCurveClosed=False,
        LeafProfileCurveSymmetry=[0],
        LeafProfileCurveReverse=True,
        LeafProfileCurveControlPoints=np.array([
            [+0.0,  0.0],
            [+0.1,  0.0],
            [+0.2,  0.0],
            [+0.5,  0.1],
            [+1.0,  0.2],
        ]),
        LeafWidthXFunc='interp',
        LeafWidthXXVals=(0, 1),
        LeafWidthXYVals=np.array([
            0.09, 0.1, 0.14, 0.24, 0.29, 0.33, 0.3, 0.25, 0.18, 0
        ]),
        LeafTwistXFunc='sin',
        LeafTwistXAmplitude=(2.0 * np.pi * 0.5),
        LeafTwistXPeriod=(1.0 / 3.0),
        LeafBendXMethod='LeafBendX',
        InternodeWidthNFunc='linear',
        InternodeWidthNSlope=-0.5,
        InternodeWidthNIntercept=0.9,
        BranchAngleNFunc='linear',
        BranchAngleNSlope=-0.4,
        BranchAngleNIntercept=0.5,
    )
    _leaf_data_parameters = [
        'Length', 'Width', 'Area',
    ]
    _length_parameters = [
        'Length', 'Width',
    ]
    _area_parameters = [
        'Area',
    ]
    _cached_leaf_data = {}
    _attribute_properties = [
        'leaf_data_file', 'leaf_data_units',
        'unfurl_leaves', 'crop_class',
    ]

    def __init__(self, **kwargs):
        self._leaf_data = None
        self._leaf_data_analysis = None
        super(MaizeGenerator, self).__init__(**kwargs)
        if self.leaf_data_file:
            for k in self.leaf_data_parameters:
                self.update_leaf_param(k)
        if self.unfurl_leaves:
            self.update_curve_param(
                'LeafProfile', 'circle',
                patch_param={'Var': 'X'},
            )

    # def LeafBendPath(self, age, n, x):
    #     points = np.array([
    #         [-0.5, 0],
    #         [-0.223915, 0.114315],
    #         [0.121756, 0.0370409],
    #         [0.467244, -0.216243],
    #     ])
    #     points[:, 0] = np.linspace(0, 1, points.shape[0])
    #     points[:, 1] = 0.0
    #     return CurvePlantParameter.create_curve(points)

    @classmethod
    def load_leaf_data(cls, fname, crop_class=None):
        r"""Load leaf data from a file, caching it for future use.

        Args:
            fname (str): Path to the data file.
            crop_class (str, optional): Crop class that should be
                selected by filtering the rows in the data file based
                on the values in the 'Class' column.

        Returns:
            pandas.DataFrame: Loaded data.

        """
        key = (fname, crop_class)
        if key not in cls._cached_leaf_data:
            if crop_class is None:
                print(f"Loading leaf data from \"{fname}\"")
                cls._cached_leaf_data[key] = pd.read_csv(fname)
            else:
                df0 = cls.load_leaf_data(fname)
                print(f"Selecting {crop_class} from \"{fname}\"")
                df = df0.loc[df0['Class'] == crop_class]
                if df.empty:
                    print(f"No data found for crop_class \"{crop_class}\"")
                    print(df0)
                    pdb.set_trace()
                cls._cached_leaf_data[key] = df
        return cls._cached_leaf_data[key]

    @property
    def leaf_data_parameters(self):
        r"""list: Parameters that can be read from leaf_data_file."""
        return [f'Leaf{k}' for k in self._leaf_data_parameters]

    @property
    def leaf_data(self):
        r"""pandas.DataFrame: Data contained in the leaf_data_file"""
        if self._leaf_data is None:
            if not self.leaf_data_file:
                raise AttributeError("No leaf data provided")
            self._leaf_data = self.load_leaf_data(
                self.leaf_data_file,
                crop_class=self.crop_class,
            )
            if self.length_units and self.leaf_data_units:
                length_scale = scale_factor(self.leaf_data_units,
                                            self.length_units)
                for k in self._length_parameters:
                    v = self.select_leaf_data(df=self._leaf_data,
                                              parameter=k)
                    v *= length_scale
                for k in self._area_parameters:
                    v = self.select_leaf_data(df=self._leaf_data,
                                              parameter=k)
                    v *= length_scale * length_scale
        return self._leaf_data

    @property
    def leaf_data_analysis(self):
        r"""dict: Parameters describing the leaf data."""
        if self._leaf_data_analysis is not None:
            return self._leaf_data_analysis
        nmin = 1
        nmax = 1
        self.select_leaf_data(n=nmax)
        while not self.select_leaf_data(n=nmax).empty:
            nmax += 1
        self._leaf_data_analysis = {
            'nmin': nmin,
            'nmax': nmax,
            'nvals': np.array(range(nmin, nmax)),
            'params': [],
            'dists': {},
            'dist_param': {}
        }
        for col in self.leaf_data.filter(regex=r'^V\d+ '):
            p = col.split(' ', 1)[-1]
            if p not in self._leaf_data_analysis['params']:
                self._leaf_data_analysis['params'].append(p)
        for p in self._leaf_data_analysis['params']:
            k = f'Leaf{p.title()}'
            kprops = self.get(f'{k}Dist', {}, return_other='parameters')
            profile = kprops.get('', 'normal')
            self._leaf_data_analysis['dists'][k] = profile
            df = self.select_leaf_data(parameter=p)
            self._leaf_data_analysis['dist_param'][k] = np.array(
                self.parametrize_dist(df, profile=profile, axis=0)
            ).T
        self.log(f'Leaf data analysis:\n'
                 f'{pprint.pformat(self._leaf_data_analysis)}')
        return self._leaf_data_analysis

    def select_leaf_data(self, df=None, crop_class=None,
                         parameter=None, n=None):
        r"""Select a subset of leaf data.

        Args:
            df (pandas.DataFrame, optional): Data frame that should be
                filtered instead of self.leaf_data.
            crop_class (str, optional): Crop class that should be selected
            parameter (str, optional): Parameter that should be selected.
            n (int, optional): Phytomer count that should be selected.

        Returns:
            pandas.DataFrame: Selected data.

        """
        if df is None:
            df = self.leaf_data
        if crop_class is not None:
            df = df.loc[df['Class'] == crop_class]
        if parameter is not None:
            df = df.filter(regex=f' {parameter.title()}$')
        if n is not None:
            assert not (n % 1)
            n = int(n) + 1
            df = df.filter(regex=f'^V{n} ')
        return df

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
            mean = np.nanmean(values, **kwargs)
            std = np.nanstd(values, **kwargs)
            param = (mean, std)
        elif profile in ['choice']:
            param = (values, )
        else:
            raise ValueError(f"Unsupported profile \"{profile}\"")
        return param

    def update_leaf_param(self, k):
        r"""Update a leaf parameter to use data from leaf_data_file.

        Args:
            k (str): Leaf data parameter to update.

        """
        v = self.get(k, None, return_other='instance')
        if v is None:
            self.log(f'No leaf parameter \"{k}\"')
            return
        assert k in self.leaf_data_analysis['dists']
        v.parameters[''] = 1.0
        remove = ['Func', 'Dist', 'NFunc']
        for kr in remove:
            vr = v.parameters.get(kr, None)
            if isinstance(vr, PlantParameterBase):
                vr.clear()
            else:
                v.parameters.pop(kr, None)
        v.parameters['NFunc'].update({
            f'{k}NFunc': 'user',
            f'{k}NFunction': self.leaf_data_function(k),
            f'{k}NMax': 1.0,
        })
        v.log(f"{k}:\n{pprint.pformat(v.contents)}")

    def update_curve_param(self, k, other_param, other_is_end=False,
                           patch_param=None):
        r"""Update a curve parameter to use a patch.

        Args:
            k (str): Curve parameter.
            other_param (dict): Parameters for new curve use to create
                the patch.
            other_is_end (bool, optional): If True, the existing curve
                is treated as the starting curve and the other_param are
                used for the end curve.
            patch_param (dict, optional): Patch specific parameters that
                should be used for the new patch (with non-prefixed
                patch parameters as keys).

        """
        curve_base = f'{k}Curve'
        patch_base = f'{k}CurvePatch'
        curve = self.get(curve_base, return_other='instance')
        exist_param = copy.deepcopy(curve.parameters)
        if other_is_end:
            other_curve_base = f'{patch_base}EndCurve'
            exist_curve_base = f'{patch_base}StartCurve'
        else:
            other_curve_base = f'{patch_base}StartCurve'
            exist_curve_base = f'{patch_base}EndCurve'
        if other_param == 'circle':
            other_param = copy.deepcopy(exist_param)
            y = np.linspace(-1, 1, other_param['ControlPoints'].shape[0])
            x = np.sqrt(1.0 - y**2)
            other_param['ControlPoints'][:, 0] = x
            other_param['ControlPoints'][:, 1] = y
        curve.clear()
        param = {}
        for k in CurvePlantParameter._patch_properties:
            if k in exist_param:
                param[f'{curve_base}{k}'] = exist_param.pop(k)
        if patch_param:
            for k, v in patch_param.items():
                param[f'{patch_base}{k}'] = v
        for k, v in exist_param.items():
            param[f'{exist_curve_base}{k}'] = v
        for k, v in other_param.items():
            param[f'{other_curve_base}{k}'] = v
        curve.update(param)
        curve.debug(f'CURVE:\n{pprint.pformat(curve.contents)}',
                    force=True)

    def leaf_data_function(self, k):
        r"""Get a function that samples the distribution of parameters
        for a given n.

        Args:
            k (str): Leaf data parameter that should be sampled.

        Returns:
            callable: Function.

        """
        profile = self.leaf_data_analysis['dists'][k]
        nvals = self.leaf_data_analysis['nvals']
        param = self.leaf_data_analysis['dist_param'][k]
        f = scipy.interpolate.interp1d(nvals, param, axis=0)

        def leaf_function(n):
            return self.sample_dist(profile, *f(n))

        return leaf_function

    def LeafBendX(self, x):
        r"""Explicit method to compute the dependence of LeafBend on
        position along the leaf.

        Args:
            x (float, optional): Position along the leaf that will be
                generated.

        Returns:
            object: Parameter value.

        """
        amp = 0.05
        period = 0.9
        slope = 0.8
        out = amp * (np.cos(2.0 * np.pi * x / period) - 1.0) + (slope * x)
        return 2.0 * np.pi * out


def extract_lpy_param(args):
    r"""Extract parameters for the LPy system from the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: LPy parameters.

    """
    # TODO: Remove this one parameters are finalized
    args.overwrite_lpy_param = True
    if args.lpy_param and (os.path.isfile(args.lpy_param)
                           and not args.overwrite_lpy_param):
        with open(args.lpy_param, 'r') as fd:
            out = rapidjson.load(fd)
        return out
    generator_args = {
        'leaf_data_file': args.leaf_data,
        'leaf_data_time': args.leaf_data_time,
        'leaf_data_units': args.leaf_data_units,
        'crop_class': args.crop_class,
        'unfurl_leaves': args.unfurl_leaves,
        'verbose': args.verbose,
        'debug': args.debug,
        'debug_param': args.debug_param,
    }
    out = {
        'MAIZE3D_PARAM': generator_args,
        'NLEAFDIVIDE': args.n_leaf_divide,
        'OUTPUT_TIME': args.age.value,
    }
    if args.lpy_param:
        with open(args.lpy_param, 'w') as fd:
            rapidjson.dump(out, fd)
    return out


############################################################
# TASKS
############################################################

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
        'crop', 'crop_class', 'canopy', 'color',
        'overwrite_lpy_param', 'plantid', 'debug_param',
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
        (('--leaf-data', ), {
            'type': str,
            'help': 'File containing raw leaf data',
        }),
        # TODO: Correct this for the fact that the actual data is
        # collected when each leaf is mature
        (('--leaf-data-time', ), {
            'type': parse_quantity, 'default': 27, 'units': 'days',
            'help': ('Time that data containing raw leaf data was '
                     'collected (in days since planting)'),
        }),
        (('--leaf-data-units', ), {
            'type': parse_units, 'default': 'cm',
            'help': ('Units that lengths are in within the leaf data '
                     'file'),
        }),
        (('--n-leaf-divide', ), {
            'type': int, 'default': 10,
            'help': 'Number of segments to divide leaves into',
        }),
        # TODO: Use this
        (('--crop', ), {
            'type': str, 'choices': ['maize'], 'default': 'maize',
            'help': 'Crop to generate a geometry for.',
        }),
        (('--crop-class', ), {
            'type': str, 'choices': ['WT', 'rdla', 'all', 'all_split'],
            'default': 'WT',
            'help': 'Class to generate geometry for',
        }),
        (('--lpy-input', ), {
            'type': str,
            # TODO: Make this more generic or lookup from crop name
            'default': os.path.join(utils._lpy_dir, 'maize.lpy'),
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
            'help': 'Parameter(s) that debug mode should be enabled for.',
        }),
        (('--unfurl-leaves', ), {
            'action': 'store_true',
            'help': 'Start leaves as cylinders and then unfurl them',
        }),
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
        if args.crop_class == 'all_split':
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
        if (not args.leaf_data) and args.crop == 'maize':
            # TODO: Prevent this from being hard coded
            args.leaf_data = utils._leaf_data
        if args.color_str == 'plantids':
            args.plantids_in_blue = True
        if args.lpy_param is None:
            args.lpy_param = os.path.join(
                utils._param_dir, f'param_{args.crop_class}.json')
        if not args.crop_class.startswith('all'):
            if isinstance(args.lpy_param, str):
                from canopy_factory.crops.base import extract_lpy_param
                args.lpy_param_str = args.lpy_param
                args.lpy_param = extract_lpy_param(args)

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
        return f'{args.crop}_{args.crop_class}'

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
        if args.unfurl_leaves:
            suffix += '_unfurled'
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

    @readonly_cached_property
    def all_crop_classes(self):
        r"""list: All crop classes for current data."""
        # TODO: Get from maize module/generic crop module
        from canopy_factory.crops.base import MaizeGenerator
        df = MaizeGenerator.load_leaf_data(self.args.leaf_data)
        return sorted(list(set(df['Class'])))

    @classmethod
    def _run(cls, self):
        r"""Run the process associated with this subparser."""
        if self.args.crop_class == 'all_split':
            plantid = self.args.plantid
            x = self.args.x
            y = self.args.y
            for i, crop_class in enumerate(self.all_crop_classes):
                cls.run_class(
                    self, dont_reset_alternate_output=True,
                    dont_load_existing=True,
                    args_overwrite={
                        'x': x, 'y': y, 'plantid': plantid,
                        'crop_class': crop_class,
                        'lpy_param': None,
                    },
                )
            mesh = None
            self.add_alternate_output('plantids', None)
        elif self.args.crop_class == 'all':
            mesh = None
            plantids = self.pop_alternate_output('plantids', None)
            if plantids is None:
                plantids = []
            else:
                plantids = [plantids]
            plantid = self.args.plantid
            x = self.args.x
            y = self.args.y
            for i, crop_class in enumerate(self.all_crop_classes):
                print("IMESH", i, crop_class)
                imesh = cls.run_class(
                    self, dont_reset_alternate_output=True,
                    args_overwrite={
                        'x': x, 'y': y, 'plantid': plantid,
                        'crop_class': crop_class,
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
        self.args.lpy_param['MAIZE3D_PARAM']['seed'] = plantid
        color = self.args.color
        if self.args.color_str == 'plantids':
            color = [0, 255, plantid]
        lsys = Lsystem(self.args.lpy_input, self.args.lpy_param)
        tree = lsys.axiom
        for i in range(self.args.niter):
            tree = lsys.iterate(tree, 1)
        lsys_units = lsys.context().globals()['generator'].length_units
        scene = lsys.sceneInterpretation(tree)
        mesh = utils.scene2geom(
            scene, self.args.mesh_format,
            axis_up=self.args.axis_up,
            axis_x=self.args.axis_rows,
            color=color,
        )
        mesh = utils.scale_mesh(mesh, 1.0,
                                from_units=lsys_units,
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
