import os
import pdb
import copy
import pprint
import numpy as np
import pandas as pd
import functools
import warnings
import subprocess
import traceback
import contextlib
import re
import glob
from canopy_factory.config import PackageConfig
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping
from datetime import timedelta, datetime
import pytz
import yggdrasil_rapidjson as rapidjson
from yggdrasil_rapidjson import units
from yggdrasil_rapidjson.geometry import Ply as PlyDict
from yggdrasil_rapidjson.geometry import ObjWavefront as ObjDict


functools_cached_property = getattr(functools, "cached_property", None)
_source_dir = os.path.abspath(os.path.dirname(__file__))
cfg = PackageConfig(
    'canopy_factory',
    defaults={
        'directories': {
            'yamls': os.path.join(os.getcwd(), 'yamls'),
            'input': os.path.join(os.getcwd(), 'input'),
            'output': os.path.join(os.getcwd(), 'output'),
            'user_param': os.path.join(os.getcwd(), 'output', 'param'),
        },
    },
)
cfg.setdefaults(
    directories={
        'source': _source_dir,
        'test_output': os.path.join(
            os.path.dirname(_source_dir), 'tests', 'data'),
        'data': os.path.join(_source_dir, 'data'),
        'lpy': os.path.join(_source_dir, 'data', 'lpy'),
        'param': os.path.join(_source_dir, 'param'),
    },
    files={
        'locations': os.path.join(_source_dir, 'data', 'locations.csv'),
        'testdata': [],
    },
)
_default_axis_up = np.array([0, 0, 1], dtype=np.float64)
_default_axis_x = np.array([1, 0, 0], dtype=np.float64)
_mesh_format = 'triangle_mesh'
_geom_classes = {
    'ply': PlyDict,
    'obj': ObjDict,
    _mesh_format: PlyDict,
}
_supported_3d_formats = sorted(_geom_classes.keys())
_geom_ext = {
    '.ply': 'ply',
    '.obj': 'obj',
    '.mesh': _mesh_format,
}
_inv_geom_ext = {v: k for k, v in _geom_ext.items()}
_axis_map = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class NoDefault(object):
    r"""Stand-in for identifying if a default is passed."""
    pass


class ClassRegistry(object):
    r"""A place to register classes."""

    def __init__(self):
        self._base_registry = {}
        self._registry = {}
        self._properties = {}

    def register(self, cls):
        r"""Register a class.

        Args:
            cls (type): Class to register.

        """
        if hasattr(cls, '_on_registration'):
            cls._on_registration(cls)
        key = getattr(cls, '_registry_key', None)
        if key is None:
            return
        name = getattr(cls, '_registry_name', None)
        if name is None:
            name = getattr(cls, '_name', None)
        if name is None:
            if key in self._base_registry:
                # raise AssertionError(
                #     f'Duplicate base ({cls}) registered '
                #     f'as {key} (existing is {self._base_registry[key]})'
                # )
                return
            self._base_registry[key] = cls
        else:
            self._registry.setdefault(key, {})
            if name in self._registry[key]:
                raise AssertionError(f'Duplicate {name} registered '
                                     f'as {key}')
            self._registry[key][name] = cls

    @classmethod
    def _class2str(self, cls):
        return str(cls).rsplit('.', 1)[-1].split("\'")[0]

    @classmethod
    def _instance2str(self, instance, cls=None):
        if isinstance(instance, str):
            classname = instance
        else:
            if cls is None:
                cls = type(instance)
            classname = self._class2str(cls)
        return classname

    def registry_cached_properties(self, instance, cls=None):
        r"""Get the registry of cached properties for an instance.

        Args:
            instance (object): Registered class instance to get the
                registry for.
            cls (type, optional): Base class that registry should be
                returned for.

        """
        classname = self._instance2str(instance, cls=cls)
        if classname not in self._properties:
            self._properties[classname] = {
                'readonly': [], 'readwrite': [], 'args': []
            }
        return self._properties[classname]

    def register_cached_property(self, method, args=None,
                                 readonly=False, classname=None):
        r"""Register a cached property.

        Args:
            method (function): Method being registered.
            args (bool, optional): If True, the property will be reset
                when the args for the class are updated.
            readonly (bool, optional): If True, the property can only
                be read, not set.
            classname (str, optional): Unique name that should be used to
                register the property in place of the classname from the
                method's __qualname__.

        """
        if classname is None:
            classname, methodname = method.__qualname__.rsplit('.', 1)
        else:
            methodname = method.__qualname__.rsplit('.', 1)[-1]
        registry = self.registry_cached_properties(classname)
        if args:
            registry['args'].append(methodname)
        if readonly:
            dest = registry['readonly']
        else:
            dest = registry['readwrite']
        if methodname in dest:
            print(classname, methodname)
        assert methodname not in dest
        dest.append(methodname)

    def clear_cached_properties(self, instance, exclude=None,
                                include=None, cls=None, args=False):
        r"""Clear the cached properties for an instance.

        Args:
            instance (object): Registered class instance to clear the
                cached properties of.
            exclude (list, optional): Set of cached properties that
                should be preserved.
            include (list, optional): Subset of cached properties that
                should be cleared.
            cls (type, optional): Base class that cached properties
                should be cleared for.
            args (bool, optional): If true, only args cached properties
                should be cleared.

        """
        if exclude is None:
            exclude = []
        registry = self.registry_cached_properties(instance, cls=cls)
        for k in registry['readwrite']:
            if ((k in exclude or (args and k not in registry['args'])
                 or (include is not None and k not in include))):
                continue
            if k in instance.__dict__:
                delattr(instance, k)
        for k in registry['readonly']:
            if ((k in exclude or (args and k not in registry['args'])
                 or (include is not None and k not in include))):
                continue
            instance._cached_properties.pop(k, None)
        if cls is None:
            for base in instance._registered_base_classes:
                self.clear_cached_properties(
                    instance, exclude=exclude, include=include,
                    cls=base, args=args,
                )

    def has_cached_property(self, instance, k, cls=None):
        r"""Check if a cached property has been initialized.

        Args:
            instance (object): Registered class instance to check the
                cached properties of.
            k (str): Name of the cached property to check.
            cls (type, optional): Base class that cached properties
                should be returned for.

        Returns:
            bool: True if the cached property has been initialized.

        """
        registry = self.registry_cached_properties(instance, cls=cls)
        if k in registry['readwrite']:
            return (k in instance.__dict__)
        if k in registry['readonly']:
            return (k in instance._cached_properties)
        if cls is None:
            for base in instance._registered_base_classes:
                if self.has_cached_property(instance, k, cls=base):
                    return True
        return False

    def get_cached_properties(self, instance, exclude=None, include=None,
                              cls=None):
        r"""Get the set of cached properties for an instance.

        Args:
            instance (object): Registered class instance to get the
                cached properties of.
            exclude (list, optional): Set of cached properties that
                should not be included in the returned dictionary.
            include (list, optional): Set of cached properties that
                should be included in the returned dictionary.
            cls (type, optional): Base class that cached properties
                should be returned for.

        Returns:
            dict: Cached properties.

        """
        if exclude is None:
            exclude = []
        registry = self.registry_cached_properties(instance, cls=cls)
        out = {}
        for k in registry['readwrite']:
            if k in exclude or (include is not None
                                and k not in include):
                continue
            if k in instance.__dict__:
                out[k] = instance.__dict__[k]
        for k in registry['readonly']:
            if k in exclude or (include is not None
                                and k not in include):
                continue
            if k in instance._cached_properties:
                out[k] = instance._cached_properties[k]
        if cls is None:
            for base in instance._registered_base_classes:
                out.update(self.get_cached_properties(
                    instance, exclude=exclude, include=include, cls=base
                ))
        return out

    def set_cached_properties(self, instance, properties, cls=None):
        r"""Set cached properties.

        Args:
            instance (object): Registered class instance to set the
                cached properties of.
            properties (dict): Cached properties to update.
            cls (type, optional): Base class that cached properties
                should be set for.

        """
        registry = self.registry_cached_properties(instance, cls=cls)
        for k in registry['readwrite']:
            if k not in properties:
                continue
            setattr(instance, k, properties[k])
        for k in registry['readonly']:
            if k not in properties:
                continue
            instance._cached_properties[k] = properties[k]
        if cls is None:
            for base in instance._registered_base_classes:
                self.set_cached_properties(
                    instance, properties, cls=base
                )

    def registry(self, key=None):
        r"""Get the registry dictionary.

        Args:
            key (str, optional): Key for sub-registry that should be
                returned.

        Returns:
            dict: Registry.

        """
        if key is None:
            return self._registry
        return self._registry.get(key, {})

    def keys(self, key=None):
        r"""Get the registry keys.

        Args:
            key (str, optional): Key for sub-registry that keys should be
                returned for.

        Returns:
            iterable: Registry keys.

        """
        return self.registry(key).keys()

    def values(self, key=None):
        r"""Get the registry values.

        Args:
            key (str, optional): Key for sub-registry that values should
                be returned for.

        Returns:
            iterable: Registry values.

        """
        return self.registry(key).values()

    def items(self, key=None):
        r"""Get the registry items.

        Args:
            key (str, optional): Key for sub-registry that items should
                be returned for.

        Returns:
            iterable: Registry key/value pairs.

        """
        return self.registry(key).items()

    def getbase(self, key, default=NoDefault):
        r"""Get a base class for a registry key.

        Args:
            key (str): Sub-registry that should be accessed.
            default (type, optional): Value that should be returned if
                the requested entry is not present.

        Returns:
            type: Registry entry.

        Raises:
            KeyError: If the requested entry is not present and a default
                is not provided.

        """
        out = self._base_registry.get(key, default)
        if out is NoDefault:
            raise KeyError(key)
        return out

    def get(self, key, name, default=NoDefault):
        r"""Get a registry entry.

        Args:
            key (str): Sub-registry that should be accessed.
            name (str): Name of entry that should be returned.
            default (type, optional): Value that should be returned if
                the requested entry is not present.

        Returns:
            type: Registry entry.

        Raises:
            KeyError: If the requested entry is not present and a default
                is not provided.

        """
        out = self.registry(key).get(name, default)
        if out is NoDefault:
            raise KeyError((key, name))
        return out


_class_registry = ClassRegistry()


def get_class_registry():
    return _class_registry


def readonly_cached_property(method, args=None, **kwargs):
    r"""Read-only cached property decorator.

    Args:
        method (function): Method who's output should be cached.
        args (bool, optional): If True, the property will be reset when
            the args for the class are updated.
        **kwargs: Additional keyword arguments are passed to
            _class_registry.register_cached_property.

    Returns:
        function: Decorated method.

    """
    _class_registry.register_cached_property(method, args=args,
                                             readonly=True, **kwargs)
    methodname = method.__qualname__.rsplit('.', 1)[-1]

    @property
    def _method_wrapper(self):
        if methodname not in self._cached_properties:
            with self.calculating(methodname):
                self._cached_properties[methodname] = method(self)
        return self._cached_properties[methodname]

    return _method_wrapper


def cached_property(method, args=None, classname=None):
    r"""Cached property decorator.

    Args:
        method (function): Method who's output should be cached.
        args (bool, optional): If True, the property will be reset when
            the args for the class are updated.
        classname (str, optional): Unique name that should be used to
            register the property in place of the classname from the
            method's __qualname__.

    Returns:
        function: Decorated method.

    """
    if functools_cached_property is None:
        return readonly_cached_property(method, args=args,
                                        classname=classname)
    _class_registry.register_cached_property(method, args=args,
                                             classname=classname)
    return functools_cached_property(method)


def cached_args_property(method):
    r"""Cached property decorator that should be reset when arguments
    are modified.

    Args:
        method (function): Method who's output should be cached.

    Returns:
        function: Decorated method.

    """
    return cached_property(method, args=True)


def cached_factory_property(classname):
    r"""Cached property decorator for inside a class factory.

    Args:
        classname (str): Unique name that should be used to register the
            property for the factory produced class.

    Returns:
        function: Decorator for method.

    """
    def _cached_factory_property(method):
        return cached_property(method, classname=classname)

    return _cached_factory_property


def readonly_cached_args_property(method):
    r"""Read-only cached property decorator.

    Args:
        method (function): Method who's output should be cached.

    Returns:
        function: Decorated method.

    """
    return readonly_cached_property(method, args=True)


class RegisteredMetaClass(type):
    r"""Metaclass for registering classes."""

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        _class_registry.register(cls)
        return cls


class RegisteredClassBase(object, metaclass=RegisteredMetaClass):
    r"""Base class for classes that should be registered."""

    _name = None
    _registry_key = None
    _registry_name = None
    _registered_base_classes = []

    def __init__(self):
        self._cached_properties = {}
        self._in_calculation = []

    @contextlib.contextmanager
    def calculating(self, name):
        r"""Context for handling circular calculation dependencies.

        Args:
            name (str): Name of the property being calculated.

        """
        assert name not in self._in_calculation
        self._in_calculation.append(name)
        try:
            yield
        finally:
            self._in_calculation.pop(-1)

    def check_calculating(self, name):
        r"""Check if a variable is being calculated.

        Args:
            name (str): Name of the property to check.

        Returns:
            bool: True if the variable is being calculated.

        """
        return (name in self._in_calculation)

    @staticmethod
    def _on_registration(cls):
        import inspect
        base = inspect.getmro(cls)[1]
        if getattr(base, '_registered_base_classes', None) is not None:
            cls._registered_base_classes = (
                base._registered_base_classes + [base])

    @classmethod
    def log_class(cls, message='', debug=False, **kwargs):
        r"""Emit a log message.

        Args:
            message (str, optional): Log message.
            debug (bool, optional): If True, set a Python debugger
                breakpoint after emitting the message.
            **kwargs: Additional keyword arguments are passed to the
                format_log method.

        Returns:
            str: Log message.

        """
        msg = cls.format_log(message=message, **kwargs)
        print(msg)
        if debug:
            pdb.set_trace()
        return msg

    @classmethod
    def format_log(cls, message='', prefix=None, border=False,
                   source=None, exception=None):
        r"""Format a log message.

        Args:
            message (str, optional): Log message.
            prefix (str, optional): Prefix to use.
            border (bool, optional): If True, add a border before and
                after the message.
            source (str, optional): Source class that the message was
                emitted by.
            exception (BaseException, optional): Error that traceback
                should be reported for.

        """
        if prefix is None:
            if source is None:
                prefix = f'{cls._name}: '
            else:
                prefix = f'{cls._name} [{source}]: '
        msg = f'{prefix}{message}' if isinstance(prefix, str) else message
        if exception is not None:
            msg += '\n' + ''.join(traceback.format_exception(exception))
        if border:
            line = 80 * '-'
            msg = line + '\n' + msg + '\n' + line
        return msg

    @property
    def log_prefix(self):
        r"""str: Prefix to add to messages emitted by this instance."""
        return None

    def log(self, message='', force=False, cls=None, **kwargs):
        r"""Emit a log message.

        Args:
            message (str, optional): Log message.
            force (bool, optional): If True, print the log message even
                if self.verbose is False.
            **kwargs: Additional keyword arguments are passed to the
                log_class method.

        """
        if not (getattr(self, 'verbose', True) or force):
            return
        kwargs.setdefault('prefix', self.log_prefix)
        source = None
        if cls is None:
            cls = self
        else:
            source = self._name
        return cls.log_class(message=message, source=source, **kwargs)

    def error(self, error_cls, message='', **kwargs):
        r"""Raise an error, adding context to the message.

        Args:
            error_cls (type): Error class.
            message (str, optional): Error message.
            **kwargs: Additional keyword arguments are passed to
                log.

        """
        kwargs['force'] = True
        msg = self.log(f'ERROR: {error_cls}({message})', **kwargs)
        raise error_cls(msg)

    def debug(self, message='', **kwargs):
        r"""Set a pdb break point if debugging is active.

        Args:
            message (str, optional): Log message to show before setting
                break point.
            **kwargs: Additional keyword arguments are passed to the
                log method if it is called.

        """
        kwargs.update(
            debug=True,
            force=True,
        )
        self.log(f'DEBUG: {message}', **kwargs)

    def clear_cached_properties(self, exclude=None, include=None,
                                args=False):
        r"""Clear the cached properties.

        Args:
            exclude (list, optional): Set of cached properties that
                should be preserved.
            include (list, optional): Subset of cached properties that
                should be cleared.
            args (bool, optional): If true, only args cached properties
                should be cleared.

        """
        _class_registry.clear_cached_properties(self, exclude=exclude,
                                                include=include,
                                                args=args)

    def has_cached_property(self, k):
        r"""Check if a cached property has been initialized.

        Args:
            k (str): Name of the cached property to check.

        Returns:
            bool: True if the cached property has been initialized.

        """
        return _class_registry.has_cached_property(self, k)

    def get_cached_properties(self, exclude=None, include=None):
        r"""Get the set of cached properties.

        Args:
            exclude (list, optional): Set of cached properties that
                should not be included in the returned dictionary.
            include (list, optional): Set of cached properties that
                should be included in the returned dictionary.

        Returns:
            dict: Cached properties.

        """
        return _class_registry.get_cached_properties(
            self, exclude=exclude, include=include)

    def pop_cached_properties(self, preserve=None, exclude=None,
                              include=None, args=False):
        r"""Clear the cached properties and return their values (before
        they are cleared).

        Args:
            preserve (list, optional): Set of cached properties that
                should not be cleared.
            exclude (list, optional): Set of cached properties that
                should not be included in the returned dictionary.
            include (list, optional): Set of cached properties that
                should be included in the returned dictionary.
            args (bool, optional): If True, only args cached properties
                should be cleared.

        Returns:
            dict: Cached properties.

        """
        out = self.get_cached_properties(exclude=exclude,
                                         include=include)
        self.clear_cached_properties(exclude=preserve, args=args)
        return out

    def set_cached_properties(self, properties):
        r"""Set cached properties.

        Args:
            properties (dict): Cached properties to update.

        """
        _class_registry.set_cached_properties(self, properties)

    def reset_cached_properties(self, properties, preserve=None,
                                args=False):
        r"""Set cached properties after clearing the existing cache.

        Args:
            properties (dict): Cached properties to update.
            preserve (list, optional): Set of cached properties that
                should not be cleared.
            args (bool, optional): If True, only args cached properties
                should be cleared.

        """
        self.clear_cached_properties(exclude=preserve, args=args)
        self.set_cached_properties(properties)


############################################################
# Methods for user input
############################################################

def input_yes_or_no(question, default=True):
    r"""Ask for user response to a yes or no question.

    Args:
        question (str): Question to ask.
        default (bool, optional): If True, the default response will be
            True if no input is provided.

    Returns:
        bool: True if the user responds in the affirmative.

    """
    default_str = ' [Y/n]' if default else ' [y/N]'
    value = input(question + default_str + ': ')
    yes_values = ['y', 'yes']
    if default:
        yes_values.append('')
    return (value.lower() in yes_values)


############################################################
# Methods for I/O
############################################################

class FilenameGenerationError(BaseException):
    r"""Error to raise when a file name cannot be generated."""
    pass


class SuffixGenerationError(FilenameGenerationError):
    r"""Error to raise when an argument cannot be converted into a
    suffix."""
    pass


def generate_filename(basefile, ext=None, prefix=None,
                      suffix=None, directory=None):
    r"""Generate a filename using the base name from another file.

    Args:
        basefile (str): Base file name that new file should be based on.
        ext (str, optional): File extension that should be used for the
            generated file name.
        prefix (str, optional): Prefix that should be added to the base
            file name in the generated file name.
        suffix (str, optional): Suffix that should be added to the base
            file name in the generated file name.
        directory (str, optional): Directory that the generated file name
            should be in if different than the base file name.

    Returns:
        str: New file name.

    """
    assert ext or suffix or directory
    if basefile:
        out = basefile
    elif prefix:
        out = prefix
        prefix = None
    elif suffix:
        out = suffix
        suffix = None
    else:
        raise ValueError("Cannot construct a file name without a "
                         "base file, prefix, or suffix")
    if ext:
        out = os.path.splitext(out)[0] + ext
    if suffix:
        out = suffix.join(os.path.splitext(out))
    if prefix:
        parts = os.path.split(out)
        out = os.path.join(parts[0], prefix + parts[1])
    if directory:
        out = os.path.join(directory, os.path.basename(out))
    return out


def get_3D_format(fname, dont_raise=False):
    r"""Determine the format of a 3D geometry file by inspecting the
    extension of a file.

    Args:
        fname (str): Filename to inspect.
        dont_raise (bool, optional): If True, don't raise an error if
            the format cannot be determined.

    Returns:
        str: Name of the format that the extension indicates.

    Raises:
        ValueError: If the format cannot be determined and dont_raise is
            False.

    """
    out = _geom_ext.get(os.path.splitext(fname)[-1], None)
    if out is None:
        raise ValueError(f"Could not determine a 3D geometry file "
                         f"format based on the extension \"{fname}\"")
    return out


def verify_3D_format(fname, file_format):
    r"""Check a 3D geometry files extension against a file_format.

    Args:
        fname (str): Filename to inspect.
        file_format (str): 3D geometry format to check against.

    """
    file_format_ext = get_3D_format(fname, dont_raise=True)
    if file_format_ext is not None and file_format != file_format_ext:
        warnings.warn(f'File extension for \"{fname}\" indicates a '
                      f'\"{file_format_ext}\" file, but a '
                      f'\"{file_format}\" is being written')


def write_movie(frames, fname, frame_rate=1, verbose=False):
    r"""Create a movie from a list of frames.

    Args:
        frames (list): List of image frames.
        fname (str): Name of file to write to.
        frame_rate (int, optional): Frame rate in frames per second.
        verbose (bool, optional): If True, log messages will be emitted.

    """
    if not frames:
        if verbose:
            print(f'No frames provided for creating movie \"{fname}\". '
                  f'Aborting.')
        return
    if verbose:
        print(f'Writing movie with {len(frames)} frames to \"{fname}\"'
              f':\n{pprint.pformat(frames)}')
    frame_dir = os.path.dirname(frames[0])
    fname_base = os.path.basename(os.path.splitext(fname)[0])
    fname_concat = os.path.join(frame_dir, f'concat_{fname_base}.txt')
    contents_concat = [f'file \'{os.path.basename(x)}\'' for x in frames]
    assert not os.path.isfile(fname_concat)
    with open(fname_concat, 'w') as fd:
        fd.write('\n'.join(contents_concat))
    try:
        args = ['ffmpeg']
        if not verbose:
            args += ['-loglevel', 'quiet']
        args_palette = copy.deepcopy(args)
        args_palette += [
            '-f', 'concat', '-i', os.path.basename(fname_concat),
            '-vf', 'palettegen=reserve_transparent=true', '-y',
            'palette.png',
        ]
        subprocess.check_call(args_palette, cwd=frame_dir)
        args += [
            '-r', str(frame_rate), '-f', 'concat',
            '-i', os.path.basename(fname_concat),
            '-i', 'palette.png', '-lavfi', 'paletteuse',
            fname
        ]
        if verbose:
            print(args)
        subprocess.check_call(args, cwd=frame_dir)
    finally:
        if os.path.isfile(fname_concat):
            os.remove(fname_concat)
    if verbose:
        print(f'Wrote movie with {len(frames)} frames to \"{fname}\"')


def read_csv(fname, select=None, verbose=False, include_units=True):
    r"""Read data from a CSV file.

    Args:
        fname (str): Path to file that should be read.
        select (str, list, optional): One or more fields that should be
            selected.
        verbose (bool, optional): If True, log messages will be emitted.
        include_units (bool, optional): If True, units contained in the
            header names will be added to the returned columns.

    Returns:
        dict: CSV contents with colums as key/value pairs.

    """
    if verbose:
        print(f'Reading CSV from \"{fname}\"')
    df = pd.read_csv(fname, comment='#')
    out = {}
    for k in df.columns:
        out[k] = df[k].to_numpy()
    for k in list(out.keys()):
        if ' (' in k:
            k_name, k_units = k.split(' (', 1)
            k_units = k_units.rsplit(')', 1)[0]
            if include_units:
                out[k_name] = units.QuantityArray(out.pop(k), k_units)
            else:
                out[k_name] = out.pop(k)
    if isinstance(select, str):
        out = out[select]
    elif isinstance(select, list):
        out = {k: out[k] for k in select}
    if verbose:
        print(f'Read CSV from \"{fname}\"')
    return out


def write_csv(data, fname, verbose=False, comments=None):
    r"""Write columns to a CSV file.

    Args:
        data (dict): Table columns.
        fname (str): Path to file where the CSV should be saved.
        verbose (bool, optional): If True, log messages will be emitted.
        comments (list, optional): Comments to add at the beginning of the
            file.

    """
    if verbose:
        print(f"Writing CSV to \"{fname}\"")
    header = True
    data_units = [v.units if isinstance(v, units.QuantityArray)
                  else '' for v in data.values()]
    if any(data_units):
        header = list(data.keys())
        for i, v in enumerate(data_units):
            if not v:
                continue
            header[i] += f' ({v})'
    try:
        df = pd.DataFrame(data)
    except ValueError:
        print(data)
        print(data.keys())
        print({k: type(v) for k, v in data.items()})
        pdb.set_trace()
        raise
    mode = 'w'
    if comments:
        with open(fname, 'w') as fd:
            fd.write('\n'.join([f'# {x}' for x in comments]) + '\n')
        mode = 'a'
    df.to_csv(fname, index=False, header=header, mode=mode)
    if verbose:
        print(f"Wrote CSV to \"{fname}\"")


def read_png(fname, verbose=False):
    r"""Read image data from a PNG file.

    Args:
        fname (str): Path to file the image should be read from.
        verbose (bool, optional): If True, log messages will be emitted.

    Returns:
        np.ndarray: Image data.

    """
    if verbose:
        print(f'Reading image from \"{fname}\"')
    # ImageIO version
    import imageio
    out = imageio.v3.imread(fname)
    # # PyPNG Version
    # import png
    # reader = png.Reader(filename=fname)
    # width, height, pixels, metadata = reader.asRGB8()
    # pixel_list = list(pixels)
    # rows = []
    # nvalues = metadata['planes']
    # for x in range(width):
    #     row = []
    #     for y in range(height):  # range(height - 1, -1, -1):
    #         row.append([
    #             int(pixel_list[y][x + i]) for i in range(nvalues)
    #         ])
    #     rows.append(row)
    # out = np.array(rows)
    if verbose:
        print(f'Read image from \"{fname}\"')
    return out


def write_png(data, fname, fmt=None, verbose=False):
    r"""Write image data to a file in PNG format.

    Args:
        data (np.ndarray): Image data.
        fname (str): Path to file where the image should be saved.
        verbose (bool, optional): If True, log messages will be emitted.

    """
    if verbose:
        print(f"Writing image to \"{fname}\"")
    if fmt is None:
        if data.shape[-1] == 3:
            fmt = 'RGB'
        elif data.shape[-1] == 4:
            fmt = 'RGBA'
        else:
            raise ValueError(f"Could not guess the image format based "
                             f"on the shape of the data: {data.shape}")
    data = data.astype('uint8')
    # ImageIO version
    import imageio
    imageio.imwrite(fname, data)
    # # SciPy verson
    # import scipy.misc
    # scipy.misc.imsave(fname, data)
    # # Pillow version
    # from PIL import Image
    # image = Image.new(fmt, data.shape[:2])
    # pixels = image.load()
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         pixels[i, j] = tuple(data[i, j, :])
    # image.save(fname)
    # # PyPNG Version
    # import png
    # rows = []
    # for irow in range(data.shape[0]):
    #     rows.append(data[irow, ...].flatten().tolist())
    # image = png.from_array(rows, fmt)
    # image.save(fname)
    if verbose:
        print(f"Wrote image to \"{fname}\"")


def write_3D(mesh, fname, file_format=None, dont_correct_color=False,
             mesh_units=None, file_units=None, scale=1.0, verbose=False):
    r"""Write data to a 3D geometry file.

    Args:
        mesh (ObjDict, PlyDict): 3D geometry mesh.
        fname (str): Path to file where the geometry should be saved.
        file_format (str, optional): Name of the format that the file
            should be in. If not provided, the format will be determined
            from the file extension.
        dont_correct_color (str, optional): If True, dont correct the
            vertex colors from [0, 255] integers to [0.0, 1.0] floats.
        mesh_units (str, optional): Units that mesh vertices are in.
        file_units (str, optional): Units that the mesh should be written
            in.
        scale (float, optional): Scale factor that should be applied.
        verbose (bool, optional): If True, log messages will be emitted.

    """
    if file_format is None:
        file_format = get_3D_format(fname)
    if verbose:
        print(f"Saving mesh to \"{fname}\" in \"{file_format}\" format")
    verify_3D_format(fname, file_format)
    mesh = scale_mesh(mesh, scale,
                      from_units=mesh_units,
                      to_units=file_units)
    if file_format == _mesh_format:
        columns = []
        for i in range(3):
            columns += [f'{x}{i}' for x in 'xyz']
        df = pd.DataFrame(mesh.as_mesh(), columns=columns)
        df.to_csv(fname, index=False)
    else:
        geom_cls = _geom_classes[file_format]
        if not isinstance(mesh, geom_cls):
            mesh = geom_cls(mesh)
        with open(fname, 'w') as fd:
            fd.write(str(mesh))
        if file_format == 'obj' and not dont_correct_color:
            correct_obj_color(fname)
    if verbose:
        print(f"Saved mesh to \"{fname}\" in \"{file_format}\" format")


def read_3D(fname, file_format=None, file_units=None, mesh_units=None,
            scale=1.0, verbose=False):
    r"""Read data from a 3D geometry file.

    Args:
        fname (str): Path to 3D geometry file that should be loaded. The
            file type is determined by inspecting the file extension.
        file_format (str, optional): The format that the geometry is in.
            If not provided, the format will be determined by inspecting
            the file extension.
        file_units (str, optional): Units that the mesh in the file is in.
        mesh_units (str, optional): Units that the returned mesh should be
            in.
        scale (float, optional): Scale factor that should be applied.
        verbose (bool, optional): If True, log messages will be emitted.

    Returns:
        ObjDict, PlyDict: 3D geometry mesh.

    Raises:
        RuntimeErrror: If the file does not exist.

    """
    if verbose:
        print(f'Reading 3D geometry from \"{fname}\"')
    if not os.path.isfile(fname):
        raise RuntimeError(f"File does not exists \"{fname}\"")
    if file_format is None:
        file_format = get_3D_format(fname)
    verify_3D_format(fname, file_format)
    if file_format == _mesh_format:
        geom_cls = ObjDict
        triangles = np.array(pd.read_csv(fname))
        data = geom_cls.from_mesh(triangles)
    else:
        geom_cls = _geom_classes[file_format]
        with open(fname, 'r') as fd:
            data = geom_cls(fd.read())
    data = scale_mesh(data, scale,
                      from_units=file_units,
                      to_units=mesh_units)
    if verbose:
        print(f'Read 3D geometry from \"{fname}\"')
    return data


def correct_obj_color(fname, verbose=False):
    r"""Convert colors to be floats rather than integers.

    Args:
        fname (str): Name of the file that should be corrected.
        verbose (bool, optional): If True, print log messages when the
            file does not need to be converted.

    """
    if verbose:
        print(f'Correcting colors in \"{fname}\"')
    with open(fname, 'r') as fd:
        contents = fd.read().splitlines()
    revised = []
    has_colors = False
    abort = False

    def is_int_color(v):
        vf = float(v)
        if (vf % 1) > 0:
            if verbose:
                print(f"Color value \"{v}\" is not an integer")
            return False
        if vf < 0 or vf > 255:
            if verbose:
                print(f"Color value \"{v}\" is outside expected range [0,255]")
            return False
        return True

    for x in contents:
        if not x.startswith('v '):
            revised.append(x)
            continue
        has_colors = True
        values = x.split()
        if len(values) <= 4:
            revised.append(x)
            continue
        colors = values[4:]
        if not all(is_int_color(xx) for xx in colors):
            revised.append(x)
            # abort = True
            # break
            continue  # Allow for non-homogeneous colors
        colors = [str(float(xx) / 255.0) for xx in colors]
        assert len(colors) == 3
        values = values[:4] + colors
        revised.append(' '.join(values))
    if abort:
        if verbose:
            print(f"One or more colors in \"{fname}\" did not match the "
                  f"format expected if color correction is necessary. "
                  f"The file was not corrected.")
        return
    if not has_colors:
        print(f"No vertex colors found in \"{fname}\"")
        return
    with open(fname, 'w') as fd:
        fd.write('\n'.join(revised))
    if verbose:
        print(f"Revised colors in \"{fname}\"")


def read_locations(fname=None, verbose=False):
    r"""Read location data from a CSV file.

    Args:
        fname (str, optional): Path to file containing location data. If
            not provided cfg['files']['locations'] will be used.
        verbose (bool, optional): If True, log messages will be emitted.

    Returns:
        dict: Mapping between location name and location parameters.

    """
    if fname is None:
        fname = cfg['files']['locations']
    if verbose:
        print(f'Reading location data from \"{fname}\"')
    df = pd.read_csv(fname)
    locations = sorted(list(set(df['name'])))
    out = {k: {} for k in locations}
    for k in locations:
        idf = df.loc[df['name'] == k]
        for col in idf:
            out[k][col] = np.array(idf[col])[0]
    if verbose:
        print(f'Read location data from \"{fname}\"')
    return out


############################################################
# 3D Geometry manipulation
############################################################


def xy_axes(upaxis, verbose=False):
    r"""Determine the x & y axes from the up axis.

    Args:
        upaxis (str, int): Name or index of the up axis.
        verbose (bool, optional): If True, print information about the
            transformation.

    Returns:
        tuple(2): Indices of x & y axes.

    """
    if isinstance(upaxis, str):
        upaxis = _axis_map[upaxis]
    x = (upaxis + 1) % 3
    y = (upaxis + 2) % 3
    xyz = 'xyz'
    if verbose:
        print(f"x -> {xyz[x]}")
        print(f"y -> {xyz[y]}")
        print(f"z -> {xyz[upaxis]}")
    return x, y


def create_organ_symbol(name, fname, scale=None):
    r"""Read a 3D geometry for an organ from a file.

    Args:
        name (str): Name of the organ described by the 3D geometry.
        fname (str): Path to 3D geometry file that should be loaded. The
            file type is determined by inspecting the file extension.
        scale (str, float, optional): Scaling that should be applied when
            loading the 3D geometry. If 'max' is provided, the geometry
            will be scaled such that it's maximum width along the x, y, or
            z axis is 1.0. If 'min' is provided the geometry will be
            scaled such that it's minimum width along the x, y, or z axis
            is 1.0.

    Returns:
        PlantGL.FaceSet: Geometry.

    """
    import openalea.plantgl.all as pgl
    mesh = read_3D(fname)
    mins = np.array([1e6, 1e6, 1e6])
    maxs = np.array([-1e6, -1e6, -1e6])
    for x in mesh['vertices']:
        for i in range(3):
            if x[i] > maxs[i]:
                maxs[i] = x[i]
            if x[i] < mins[i]:
                mins[i] = x[i]
    print(name, fname, mins, maxs)
    if len(mesh['faces']) == 0:
        nind = 3
    else:
        nind = 0
        for f in mesh['faces']:
            nind = max(nind, len(f))
    # Scale
    points = []
    if scale is None:
        scale = np.ones(3, 'float')
    elif scale == 'max':
        scale = np.ones(3, 'float') / max(maxs - mins)
    elif scale == 'min':
        scale = np.ones(3, 'float') / min(maxs - mins)
    else:
        scale = scale * np.ones(3, 'float') / max(maxs - mins)
    for x in mesh['vertices']:
        xarr = scale * np.array(x)
        points.append(pgl.Vector3(xarr[0], xarr[1], xarr[2]))
    points = pgl.Point3Array(points)
    if nind == 3:
        vect_class = pgl.Index3
        index_class = pgl.Index3Array
        smb_class = pgl.TriangleSet
    elif nind == 4:
        vect_class = pgl.Index4
        index_class = pgl.Index4Array
        smb_class = pgl.QuadSet
    elif nind <= 5:
        vect_class = pgl.Index
        index_class = pgl.IndexArray
        smb_class = pgl.FaceSet
    else:
        raise ValueError(f"No PlantGL class for {nind} vertex faces")
    indices = []
    for x in mesh['faces']:
        x_int = [int(_x) for _x in x]
        indices.append(vect_class(*x_int))
    indices = index_class(indices)
    smb = smb_class(points, indices)
    smb.name = name
    return smb


def prune_empty_faces(mesh, area_min=None):
    r"""Remove faces with very small areas from a mesh.

    Args:
        mesh (ObjDict): Mesh.
        area_min (float, optional): Minimum area that should be allowed.
            If not provided the resolution of np.float32 will be used.

    Returns:
        ObjDict: Mesh with the empty faces removed.

    """
    if area_min is None:
        area_min = np.finfo(np.float32).resolution
    areas = np.array(mesh.areas)
    area_mask = (areas > area_min)
    nempty = np.logical_not(area_mask).sum()
    if nempty == 0:
        return mesh
    # print(f'Removing {nempty}/{len(areas)} faces with areas <= '
    #       f'{area_min}')
    from canopy_factory.raytrace import RayTracerBase
    out = RayTracerBase.select_faces(mesh, area_mask)
    areas = np.array(out.areas)
    area_mask = (areas > area_min)
    nempty = np.logical_not(area_mask).sum()
    if nempty > 0:
        print(f'{nempty} faces with areas <= {area_min} remain')
        prune_empty_faces(out, area_min=area_min)
        pdb.set_trace()
    assert nempty == 0
    return out


def scene2geom(scene, cls, d=None, verbose=False, colormap=None,
               components=None, **kwargs):
    r"""Convert a PlantGL scene to a 3D geometry mesh.

    Args:
        scene (plantgl.Scene): Scene to convert.
        cls (type, str): Name of the type of mesh that should be returned
            or the dictionary class that should be created.
        d (plantgl.Tesselator, optional): PlantGL discretizer.
        verbose (bool, optional): If True, display log messages about
            tasks.
        colormap (str, dict, optional): Name of matplotlib colormap or
            a dictionary mapping IDs to colors for each scene ID.
        components (dict, optional): Dictionary that component face
            face ranges should be added to.
        **kwargs: Additional keyword arguments are passed to shape2dict
            for each shape in the scene.

    Returns:
        cls: 3D geometry mesh for the scene.

    """
    if d is None:
        from openalea.plantgl.all import Tesselator
        d = Tesselator()
    cls_name = None
    if isinstance(cls, str):
        cls_name = cls
        cls = _geom_classes[cls_name]
    out = cls()
    as_obj = isinstance(out, ObjDict)
    scene_dict = scene.todict()
    if verbose:
        print(f'scene2geom: Converting scene with {len(scene_dict)} '
              f'components')
    cmap = None
    if isinstance(colormap, str):
        import matplotlib as mpl
        kmax = max(scene_dict.keys())

        def fcmap(x):
            return tuple([
                int(255 * i) for i in
                mpl.colormaps['viridis'](x/kmax)[:-1]
            ])

        cmap = fcmap

    elif isinstance(colormap, dict):
        cmap = colormap.get
    for k, shapes in scene_dict.items():
        if cmap is not None:
            kwargs['color'] = cmap(k)
        for shape in shapes:
            d.clear()
            shapedict = shape2dict(shape, d=d, as_obj=as_obj,
                                   verbose=verbose, **kwargs)
            igeom = cls.from_dict(shapedict)
            if igeom is not None:
                igeom = prune_empty_faces(igeom)
                if components is not None:
                    from canopy_factory.crops.base import ColorPlantParameter
                    component = ColorPlantParameter.colorname2component(
                        shape.appearance.name)
                    components.setdefault(component, [])
                    components[component].append((out.nface, igeom.nface))
                out.append(igeom)
            d.clear()
    if verbose:
        print('scene2geom: Finished converting scene')
    return out


def shape2dict(shape, d=None, conversion=1.0, as_obj=False,
               color=None, verbose=False,
               axis_up=_default_axis_up, axis_x=_default_axis_x):
    r"""Convert a PlantGL shape to a 3D geometry dictionary.

    Args:
        shape (plantgl.Shape): Shape to convert.
        d (plantgl.Tesselator, optional): PlantGL discretizer.
        conversion (float, optional): Conversion that should be applied
            to the returned geometry.
        as_obj (bool, optional): If True, the returned dictionary should
            be compatible with ObjWavefront meshes.
        color (tuple, optional): RGB values that should be applied to
            each vertex in the shape if the shape does not have colors.
        verbose (bool, optional): If True, display log messages about
            tasks.
        axis_up (np.ndarray, optional): Direction that should be
            considered "up" for the shape. The generated positions will
            be rotated to satisfy this and axis_x.
        axis_x (np.ndarray, optional): Direction that x coordinates in
            the shape should be mapped to. The generated positions will
            be rotated to satisfy this and axis_up.

    Returns:
        dict: Dictionary of 3D geometry components.

    """
    if d is None:
        from openalea.plantgl.all import Tesselator
        d = Tesselator()
    if verbose:
        print("Tesselating shape")
    d.process(shape)
    if d.result is None:
        raise RuntimeError("Discretization failed")
    if verbose:
        print(f"Converting {len(d.result.indexList)} faces to mesh")
    out = {}
    # Vertices
    out.setdefault('vertices', [])
    vert_arr = np.zeros((len(d.result.pointList), 3), np.float64)
    for i, p in enumerate(d.result.pointList):
        # new_vert = {}
        for j, k in enumerate(['x', 'y', 'z']):
            vert_arr[i, j] = float(conversion * getattr(p, k))
    # By default the plant is generated in the z direction, but most
    # obj viewers set y and up an z as depth. Setting upaxis to "y"
    # rotates the axes to match that expected by viewers such that the
    # solar ray tracer can consider east to be x and north to be y.
    if not (np.allclose(axis_x, _default_axis_x)
            and np.allclose(axis_up, _default_axis_up)):
        R = rotate_axes(axis_x=axis_x, axis_z=axis_up)
        vert_arr = np.matmul(R, vert_arr.T).T
    for v in vert_arr:
        out['vertices'].append(
            {k: vv for k, vv in zip(['x', 'y', 'z'], v)})
    # Colors
    if d.result.colorPerVertex and d.result.colorList:
        color = None
        if d.result.isColorIndexListToDefault():
            for i, c in enumerate(d.result.colorList):
                for k in ['red', 'green', 'blue']:
                    out['vertices'][i][k] = getattr(c, k)
        else:  # pragma: debug
            raise Exception("Indexed vertex colors not supported.")
    elif color is None:
        color = tuple(getattr(shape.appearance.ambient, k)
                      for k in ['red', 'green', 'blue'])
    if color:
        for x in out['vertices']:
            for k, c in zip(['red', 'green', 'blue'], color):
                x[k] = c
    # Faces
    if as_obj:
        for i3 in d.result.indexList:
            out.setdefault('faces', [])
            out['faces'].append([{'vertex_index': int(i3[j])}
                                 for j in range(len(i3))])
    else:
        for i3 in d.result.indexList:
            out.setdefault('faces', [])
            out['faces'].append(
                {'vertex_index': [int(i3[j]) for j in
                                  range(len(i3))]})
    return out


def rotation_matrix(theta, u):
    r"""Get the rotation matrix necessary to rotate a 3D point around
    a unit vector by a specified angle.

    Args:
        theta (float): Angle to rotate by (in radians).
        u (array): Vector to rotate around.

    Returns
        np.ndarray: Rotation matrix.

    """
    norm = np.linalg.norm(u)
    assert (norm > 0)
    u = u / norm
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    inv_cos_theta = 1 - cos_theta
    R = np.array(
        [[cos_theta + (u[0] * u[0] * inv_cos_theta),
          (u[0] * u[1] * inv_cos_theta) - (u[2] * sin_theta),
          (u[0] * u[2] * inv_cos_theta) + (u[1] * sin_theta)],
         [(u[1] * u[0] * inv_cos_theta) + (u[2] * sin_theta),
          cos_theta + (u[1] * u[1] * inv_cos_theta),
          (u[1] * u[2] * inv_cos_theta) - (u[0] * sin_theta)],
         [(u[2] * u[0] * inv_cos_theta) - (u[1] * sin_theta),
          (u[2] * u[1] * inv_cos_theta) + (u[0] * sin_theta),
          cos_theta + (u[2] * u[2] * inv_cos_theta)]], dtype='f4')
    return R


def rotate_axes(axis_x=None, axis_y=None, axis_z=None, check=False):
    r"""Get the rotation matrix for going from the standard axes set to
    another. At least two input axes are required.

    Args:
        axis_x (np.ndarray, optional): New x-axis after rotation.
        axis_y (np.ndarray, optional): New y-axis after rotation.
        axis_z (np.ndarray, optional): New z-axis after rotation.
        check (bool, optional): If True, the rotation matrix should be
            checked that it produces the desired result.

    Returns:
        np.ndarray: 3D rotation matrix.

    """
    axes_unset = [(axis_x is None), (axis_y is None), (axis_z is None)]
    if sum(axes_unset) > 1:
        raise ValueError("At least two input axes are required")
    if axis_x is None:
        axis_x = np.cross(axis_y, axis_z)
    if axis_y is None:
        axis_y = np.cross(axis_z, axis_x)
    if axis_z is None:
        axis_z = np.cross(axis_x, axis_y)
    x0 = np.zeros((3,), np.float64)
    y0 = np.zeros((3,), np.float64)
    z0 = np.zeros((3,), np.float64)
    x0[0] = 1
    y0[1] = 1
    z0[2] = 1
    angle_z = np.arccos(np.dot(z0, axis_z))
    rotax_z = np.cross(z0, axis_z)
    R_z = rotation_matrix(angle_z, rotax_z)
    if check:
        zp = np.matmul(R_z, z0)
        print(f"pi/2 = {np.pi/2}")
        print(f"angle_z = {angle_z}")
        print(f"rotax_z = {rotax_z}")
        print(f"zp = {zp}")
        print(f"axis_z = {axis_z}")
        assert np.allclose(zp, axis_z)
    xp = np.matmul(R_z, x0)
    angle_x = np.arccos(np.dot(xp, axis_x))
    rotax_x = np.cross(xp, axis_x)
    R_x = rotation_matrix(angle_x, rotax_x)
    if check:
        xpp = np.matmul(R_x, xp)
        assert np.allclose(xpp, axis_x)
    R = np.matmul(R_x, R_z)
    if check:
        xp = np.matmul(R, x0)
        yp = np.matmul(R, y0)
        zp = np.matmul(R, z0)
        print(f"xp = {xp}")
        print(f"yp = {yp}")
        print(f"zp = {zp}")
        print(f"axis_x = {axis_x}")
        print(f"axis_y = {axis_y}")
        print(f"axis_z = {axis_z}")
        assert np.allclose(xp, axis_x)
        assert np.allclose(yp, axis_y)
        assert np.allclose(zp, axis_z)
    return R


def scale_factor(from_units=None, to_units=None):
    r"""Get a scale factor for going from one set of units to another.

    Args:
        from_units (str, optional): Starting units.
        to_units (str, optional): Ending units.

    Returns:
        float: Scale factor.

    """
    if from_units is None or to_units is None or from_units == to_units:
        return 1.0
    if isinstance(from_units, str):
        from_units = units.Quantity(1.0, from_units)
    return float(from_units.to(to_units))


def scale_mesh(mesh, scale, from_units=None, to_units=None):
    r"""Scale a 3D mesh.

    Args:
        mesh (ObjDict): Mesh to shift.
        scale (float): Scale factor.
        from_units (str, optional): Units that mesh is currently in.
        to_units (str, optional): Units that the mesh should be converted
            to.

    Returns:
        ObjDict: Scaled mesh.

    """
    scale *= scale_factor(from_units, to_units)
    if scale == 1.0:
        return mesh
    mesh_dict = mesh.as_array_dict()
    if isinstance(mesh, ObjDict):
        mesh_dict['face'] -= 1
    mesh_dict['vertex'] *= scale
    return type(mesh).from_array_dict(mesh_dict)


def shift_mesh(mesh, x, y, axis_up=np.array([0, 0, 1], dtype=np.float64),
               axis_x=np.array([1, 0, 0], dtype=np.float64),
               plantids_in_blue=False, plantid=None):
    r"""Shift a 3D mesh.

    Args:
        mesh (ObjDict): Mesh to shift.
        x (float): Amount to shift the plant in the x direction.
        y (float): Amount to shift the plant in the y direction.
        axis_up (np.ndarray, optional): Direction that should be
            considered "up" for the mesh. The shifts will be performed
            perpendicular to this direction.
        axis_x (np.ndarray, optional): Direction that shifts in the x
            direction should be performed. The y axis will be determined
            from the cross product of axis_up & axis_x.
        plantids_in_blue (bool, optional): If True, plant IDs are stored
            in the blue color channel for the vertices and should be
            shifted.
        plantid (int, optional): Amount that colors should be shifted in
            the blue channel to account for plant ID.

    Returns:
        ObjDict: Shifted mesh.

    """
    mesh_dict = mesh.as_array_dict()
    if isinstance(mesh, ObjDict) and 'face' in mesh_dict:
        mesh_dict['face'] -= 1
    if 'vertex' in mesh_dict:
        axis_y = np.cross(axis_up, axis_x)
        mesh_dict['vertex'] += x * axis_x
        mesh_dict['vertex'] += y * axis_y
    if plantid and plantids_in_blue and 'vertex_colors' in mesh_dict:
        mesh_dict['vertex_colors'][:, 2] += plantid
    out = type(mesh).from_array_dict(mesh_dict)
    return out


#################################################################
# Visualization tools
#################################################################

def apply_color_map(values, color_map=None,
                    vmin=None, vmax=None, scaling='linear',
                    highlight=None, highlight_color=(255, 0, 255),
                    mask_invalid=False, include_alpha=False):
    r"""Apply a color map to a set of scalar.

    Args:
        values (arr): Scalar values that should be mapped to colors
            for each face.
        color_map (str, optional): The name of the color map that should
            be used. Defaults to 'plasma'.
        vmin (float, optional): Value that should map to the minimum of
            the colormap. Defaults to min(values).
        vmax (float, optional): Value that should map to the maximum of
            the colormap. Defaults to max(values).
        scaling (str, optional): Scaling that should be used to map the
            scalar array onto the colormap. Defaults to 'linear'.
        highlight (int, optional): Index of a value that should be
            highlighted.
        highlight_color (tuple, optional): RGB values for color that
            should be used for the highlighted value.
        mask_invalid (bool, optional): If True, mask invalid values (e.g.
            values that are <= 0 for scaling == 'log'),
        include_alpha (bool, optional): If True, include the alpha
            channel in the returned array.

    Returns:
        np.ndarray: RGB colors for each scalar in the provided array.

    """
    from matplotlib import cm
    from matplotlib import colors as mpl_colors
    from matplotlib import colormaps as mpl_colormaps
    if scaling == 'log' and mask_invalid:
        values = np.ma.MaskedArray(values, values <= 0)
    # Get color scaling
    if color_map is None:
        color_map = 'plasma'
    if vmin is None:
        if scaling == 'log' and not mask_invalid:
            vmin = values[values > 0].min()
        else:
            vmin = values.min()
        print(f"APPLY VMIN = {vmin}")
    if vmax is None:
        vmax = values.max()
        print(f"APPLY VMAX = {vmin}")
    if scaling == 'log' and not mask_invalid:
        values[values <= 0] = vmin
    # Scale colors
    if isinstance(vmin, np.ma.core.MaskedConstant):
        assert isinstance(vmax, np.ma.core.MaskedConstant)
        colors = np.zeros((values.shape[:], 3), 'int')
    else:
        cmap = mpl_colormaps[color_map]
        if scaling == 'log':
            norm = mpl_colors.LogNorm(vmin=vmin, vmax=vmax)
        elif scaling == 'linear':
            norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        else:  # pragma: debug
            raise Exception("Scaling must be 'linear' or 'log'.")
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        if include_alpha:
            max_channel = 4
        else:
            max_channel = 3
        colors = (255 * m.to_rgba(values)).astype(
            'int')[..., :max_channel]
    if highlight is not None:
        if isinstance(highlight, str):
            if highlight == 'min':
                highlight = np.argmin(values)
            elif highlight == 'max':
                highlight = np.argmax(values)
            else:
                raise ValueError(f"Unsupported highlight method "
                                 f"\"{highlight}\"")
        elif not isinstance(highlight, (int, np.ndarray, slice, tuple)):
            raise ValueError(f"Unsupported highlight method type "
                             f"{type(highlight)}")
        colors[highlight[:], :] = highlight_color[:]
    return colors


#################################################################
# Tools for parsing CLI arguments
#################################################################
# TODO: Move these into cli.py?

def jsonschema2argument(json, no_defaults=False):
    r"""Contruct a argparser argument from a jsonschema description.

    Args:
        json (dict): JSON schema.
        no_defaults (bool, optional): If True, don't include defaults.

    Returns:
        dict: Keyword arguments for adding an argument to an argument
            parser.

    """
    out = {}
    if 'oneOf' in json:
        for x in json['oneOf']:
            errors = []
            try:
                return jsonschema2argument(x)
            except BaseException as e:
                errors.append(str(e))
        else:
            raise TypeError(f'Failed to convert any of the options '
                            f'to an argument: {json["oneOf"]}, errors = '
                            f'{errors}')
    typename = json.get('subtype', json.get('type', None))
    items = json.get('items', None)
    if typename is not None:
        if ((isinstance(typename, list) and len(typename) == 2
             and 'array' in typename)):
            nonarray = [x for x in typename if x != 'array'][0]
            if items is None:
                items = dict(json, type=nonarray)
            typename = 'array'
        if not isinstance(typename, str):
            raise TypeError(f'JSON type should be string, not {typename} '
                            f'(json = {json})')
        if typename == 'array' and isinstance(items, dict):
            out = jsonschema2argument(items)
            if (('minItems' in json and 'maxItems' in json
                 and json['minItems'] == json['maxItems'])):
                out['nargs'] = json['minItems']
            else:
                out['nargs'] = '+'
        elif typename == 'boolean':
            if no_defaults:
                out.update(
                    nargs='?',
                    const=(not json.get('default', False)),
                )
            else:
                assert 'action' not in out
                if json.get('default', False):
                    out['action'] = 'store_false'
                else:
                    out['action'] = 'store_true'
        elif typename == 'string':
            out['type'] = str
        elif typename in ['number', 'float']:
            out['type'] = float
        elif typename in ['integer', 'int', 'uint']:
            out['type'] = int
        elif typename == 'function':
            out['type'] = rapidjson.loads
    if 'enum' in json:
        out['choices'] = json['enum']
    if not out:
        raise TypeError(f'Unsupported JSON schema: {json}')
    if 'default' in json and not no_defaults:
        out['default'] = json['default']
    if json.get('type', None) == 'ndarray':
        out['nargs'] = '+'
    if 'description' in json:
        out['help'] = json['description']
    return out


def parse_existing_file(x):
    r"""Parse an existing file.

    Args:
        x (str): String containing the path to an existing file.

    Returns:
        str: File name.

    """
    if x is None:
        return x
    assert isinstance(x, str) and os.path.isfile(x)
    return os.path.abspath(x)


def parse_units(x):
    r"""Parse a units string.

    Args:
        x (str): String containing units.

    Returns:
        units.Units: Units instance.

    """
    return units.Units(x)


def parse_solar_time(x):
    r"""Parse an input for an hour allowing for name-based times.

    Args:
        x (str): String containing hour.

    Returns:
        int, str: Hour.

    """
    if x in SolarModel._solar_times:
        return x
    return int(x)


def parse_solar_date(x):
    r"""Parse an input for a doy allowing for name-based times.

    Args:
        x (str): String containing doy.

    Returns:
        int, str: Day of year.

    """
    if x in SolarModel._solar_dates:
        return x
    return int(x)


def parse_datetime(x):
    r"""Parse an input that is a datetime.datetime object or a string
    representation of a datetime.datetime object that can be parsed
    via datetime.fromisoformat.

    Args:
        x (str): String containing datetime.

    Returns:
        datetime.datetime: Date time.

    """
    if isinstance(x, datetime):
        return x
    return datetime.fromisoformat(x)


class ChoiceArgument:
    r"""Wrapper for argument type that allows for named choices.

    Args:
        parser (callable): Type or argument parser that should be applied
            if the argument is not one of the named choices.
        named_choices (list, optional): Set of string values that should
            also be allowed.

    """

    def __init__(self, parser, named_choices=None):
        self.parser = parser
        self.named_choices = named_choices

    def __call__(self, x):
        if self.named_choices and x in self.named_choices:
            return x
        return self.parser(x)


class DatetimeArgument(ChoiceArgument):
    r"""Wrapper for argument type that produces a datetime.

    Args:
        named_choices (list, optional): Set of string values that should
            also be allowed.

    """

    def __init__(self, named_choices=None):
        super(DatetimeArgument, self).__init__(
            parse_datetime, named_choices=named_choices)


class QuantityArgument:
    r"""Wrapper for argument type that produces a quantity with
    the specified default units.

    Args:
        default_units (str, optional): Units that should be added to the
            returned value if x does not have units or that x should be
            converted to if it has units.
        named_choices (list, optional): Set of string values that should
            also be allowed.

    """

    def __init__(self, default_units=None, named_choices=None):
        self.default_units = default_units
        self.named_choices = named_choices

    def __call__(self, x):
        if self.named_choices and x in self.named_choices:
            return x
        return parse_quantity(x, self.default_units)


def parse_quantity(x, default_units=None):
    r"""Parse a quanity with units.

    Args:
        x (str, float, units.Quantity): Quantity value with or without
            units.
        default_units (str, optional): Units that should be added to the
            returned value if x does not have units or that x should be
            converted to if it has units.

    Returns:
        units.Quantity: Value with units.

    """
    if x is None:
        return x
    if isinstance(x, str):
        x_units = None
        if ' ' in x:
            x, x_units = x.split(' ', 1)
        try:
            x = float(x)
        except ValueError:
            if x_units is not None:
                raise
            i = 1
            while i < len(x):
                try:
                    float(x[:i])
                    i += 1
                except ValueError:
                    if i > 1:
                        i -= 1
                        x_units = x[i:]
                        x = x[:i]
                    break
            x = float(x)
        if x_units is not None:
            x = units.Quantity(x, x_units)
    if default_units is None:
        return x
    if isinstance(x, (units.Quantity, units.QuantityArray)):
        return x.to(default_units)
    elif isinstance(x, np.ndarray):
        return units.QuantityArray(x, default_units)
    return units.Quantity(x, default_units)


def parse_axis(x):
    r"""Parse an input string containing an axis tuple.

    Args:
        x (str): Input string.

    Returns:
        np.ndarray: Axis vector.

    """
    if isinstance(x, np.ndarray) or x is None:
        return x
    if x in _axis_map:
        out = np.zeros((3,), dtype=np.float64)
        out[_axis_map[x]] = 1
        return out
    if x.startswith(('[', '(')):
        x = x[1:-1]
    out = np.array([float(xx) for xx in x.split(',')], dtype=np.float64)
    assert len(out) == 3
    return out


def parse_color(x, convert_names=False):
    r"""Parse an input string containing a color tuple.

    Args:
        x (str): Input string.
        convert_names (bool, optional): If True, color names (e.g.
            'green') will be converted to their tuple version.

    Returns:
        tuple: Color tuple.

    """
    import matplotlib
    if x is None or x == 'plantid':
        return None
    special_cases = {
        'transparent': [0, 0, 0, 0],
    }
    if ',' in x:
        if x.startswith(('[', '(')):
            x = x[1:-1]
        out = tuple([int(float(x)) for x in x.split(',')])
    else:
        if x in special_cases:
            if not convert_names:
                return x
            return special_cases[x]
        out = matplotlib.colors.to_rgb(x)
        if not convert_names:
            return x
        out = [int(255 * x) for x in out]
    assert len(out) == 3
    return out


def format_list_for_help(vals, sep=', '):
    r"""Format a list in a help friendly way.

    Args:
        vals (list): Values to format.
        sep (str, optional): Separator to use.

    Returns:
        str: Formatted list.

    """
    if len(vals) > 1:
        vals = vals[:-2] + [' & '.join(vals[-2:])]
    return sep.join(vals)


def is_date(x):
    r"""Check if datetime instance is purely a date without time
    information.

    Args:
        x (datetime.datetime): Datetime instance to check.

    Returns:
        bool: True if x is purely a date.

    """
    if not isinstance(x, datetime):
        return False
    return all(getattr(x, k) == 0 for k in
               ['hour', 'minute', 'second', 'microsecond'])


def to_date(x):
    r"""Convert a datetime instance to a form that is purely a date.

    Args:
        x (datetime.datetime): Datetime instance to convert.

    Returns:
        datetime.datetime: Version with time information removed.

    """
    kws = {k: getattr(x, k) for k in
           ['year', 'month', 'day', 'tzinfo']}
    out = datetime(**kws)
    assert is_date(out)
    return out


class SolarModel(object):
    r"""Solar model using pvlib. For quantities with units, values can
    be provided as floats (in which case the default units will be
    assumed) or units.Quantity instances.

    Args:
        latitude (float): Location latitude (in degrees).
        longitude (float): Location latitude (in degrees).
        time (datetime.datetime): Time.
        altitude (float, optional): Altitude (in meters) used to compute
            solar position. If not provided, but pressure is, pressure
            will be used to calculate altitude.
        pressure (float, optional): Pressure (in Pa) used to compute
            solar position. If not provided, but altitude is, altitude
            will be used to calculate pressure.
        temperature (float, optional): Air temperature (in degrees C)
            used to compute solar position.
        eta_par (float, optional): Fraction of solar radiation (assuming
            black-body spectrum of 5800 K) that is photosynthetically
            active (wavelengths 400–700 nm).
        eta_photon (float, optional): Average number of photons per
            photosynthetically activate unit of radiation (in
            µmol s−1 W−1).
        method_solar_position (str, optional): Method that should be used
            by pvlib to determine the solar position.
        method_airmass (str, optional): Model that should be used by
            pvlib to determine the relative air mass.
        method_irradiance (str, optional): Model that should be used by
            pvlib to determine the solar irradiance.

    """

    _solar_times = ['sunrise', 'noon', 'transit', 'sunset']
    _solar_dates = [
        'summer_solstice', 'june_solstice',
        'spring_equinox', 'march_equinox',
        'winter_solstice', 'december_solstice',
        'fall_equinox', 'september_equinox',
    ]
    _season_map_northern = {
        'spring': 'march',
        'summer': 'june',
        'fall': 'september',
        'winter': 'december',
    }
    _season_map_southern = {
        'spring': 'september',
        'summer': 'december',
        'fall': 'march',
        'winter': 'june',
    }

    def __init__(self, latitude, longitude, time, altitude=None,
                 pressure=None, temperature=12.0, eta_par=0.368,
                 eta_photon=4.56, method_solar_position='nrel_numpy',
                 method_airmass='kastenyoung1989',
                 method_irradiance='ineichen'):
        import pvlib
        self.pvlib = pvlib
        if pressure is None and altitude is None:
            pressure = 101325.0
            altitude = 0.0
        self.latitude = parse_quantity(latitude, 'degrees')
        self.longitude = parse_quantity(longitude, 'degrees')
        self.time = time
        self.altitude = parse_quantity(altitude, 'meters')
        self.pressure = parse_quantity(pressure, 'Pa')
        self.temperature = parse_quantity(temperature, 'degC')
        self.eta_par = eta_par
        self.eta_photon = parse_quantity(eta_photon, 'µmol s-1 W-1')
        self.method_solar_position = method_solar_position
        self.method_airmass = method_airmass
        self.method_irradiance = method_irradiance
        if self.pressure is None:
            self.pressure = parse_quantity(pvlib.atmosphere.alt2pres(
                self.altitude.value), 'Pa')
        if self.altitude is None:
            self.altitude = parse_quantity(pvlib.atmosphere.pres2alt(
                self.pressure.value), 'meters')
        self.location = pvlib.location.Location(
            self.latitude.value, self.longitude.value,
            altitude=self.altitude.value,
            tz=self.time.tzinfo,
        )
        self.time_pv = pvlib.tools._datetimelike_scalar_to_datetimeindex(
            self.time)
        self._solar_position = None
        self._irradiance = None

    @classmethod
    def is_solar_time(cls, x):
        r"""Check if a string is a named solar time.

        Args:
            x (str): Time to check.

        Returns:
            bool: True if x is a named solar time.

        """
        return (x in cls._solar_times)

    @classmethod
    def is_solar_date(cls, x):
        r"""Check if a string is a named solar date.

        Args:
            x (str): Time to check.

        Returns:
            bool: True if x is a named solar date.

        """
        return (x in cls._solar_dates)

    def solar_time(self, x, method='spa', horizon_buffer=5.0,
                   date=None):
        r"""Parse an input string as a solar time.

        Args:
            x (str): Input string. Can be 'sunrise', 'noon', or 'sunset'.
            method (str, optional): Method that pvlib should use to
                determine the solar times.
            horizon_buffer (float, optional): Time (in minutes) that
                should be added or subtracted to times when the sun is
                at the horizon.
            date (datetime.datetime, optional): Date on which the solar
                time should be computed.

        Returns:
            datetime.datetime: Time determined from solar position.

        """
        assert x in self._solar_times
        if date is None:
            date = self.time.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif date in self._solar_dates:
            date = self.solar_date(date)
        if x == 'noon':
            x = 'transit'
        date_pv = pd.DatetimeIndex([date])
        out_pd = self.location.get_sun_rise_set_transit(
            date_pv, method=method)[x]
        out = out_pd.iloc[0].floor(freq='us').to_pydatetime()
        if horizon_buffer:
            if x == 'sunrise':
                out += quantity2timedelta(
                    parse_quantity(horizon_buffer, 'minutes'))
            elif x == 'sunset':
                out -= quantity2timedelta(
                    parse_quantity(horizon_buffer, 'minutes'))
        return out

    def solar_date(self, x, method='spa', date=None):
        r"""Parse an input string as a solar date.

        Args:
            x (str): Input string. Can be 'summer_solstice',
                'june_solstice', 'spring_equinox', 'march_equinox',
                'winter_solstice', 'december_solstice', 'fall_equinox',
                or 'september_equinox'. For those values specifying
                seasons, the latitude will be used to determine the month
                when that season occurs.
            method (str, optional): Method that pvlib should use to
                determine the solar times.
            date (datetime.datetime, optional): Date from which the year
                and timezone should be taken.

        Returns:
            datetime.datetime: Date determined from solar position.

        """
        if date is None:
            date = self.time.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        assert x in self._solar_dates
        northern = (self.latitude >= units.Quantity(0, 'degrees'))
        season_map = (
            self._season_map_northern if northern
            else self._season_map_southern
        )
        for k, v in season_map.items():
            x = x.replace(k, v)
        if x.startswith('march_'):
            month = 3
            days = [20]
            selection = 'equal'
        elif x.startswith('june_'):
            month = 6
            days = [20, 21]
            selection = 'max' if northern else 'min'
        elif x.startswith('september_'):
            month = 9
            days = [22, 23]
            selection = 'equal'
        elif x.startswith('december_'):
            month = 12
            days = [21, 22]
            selection = 'min' if northern else 'max'
        else:
            raise ValueError(x)
        dates = [
            datetime(
                year=date.year, month=month, day=day,
                tzinfo=pytz.timezone('UTC'),
            )
            for day in days
        ]
        dates_pv = pd.DatetimeIndex(dates)
        times = self.location.get_sun_rise_set_transit(
            dates_pv, method=method)
        durations = times['sunset'] - times['sunrise']
        if selection == 'equal':
            equal = pd.TimedeltaIndex(['12h'])
            idx = (durations - equal[0]).abs().values.argmin()
        elif selection == 'max':
            idx = durations.values.argmax()
        elif selection == 'min':
            idx = durations.values.argmin()
        return to_date(dates[idx].astimezone(date.tzinfo))

    @property
    def solar_position(self):
        r"""dict: Solar position information."""
        if self._solar_position is None:
            self._solar_position = self.location.get_solarposition(
                self.time_pv, pressure=self.pressure.value,
                temperature=self.temperature.value,
                method=self.method_solar_position,
            )
        return self._solar_position

    @property
    def apparent_elevation(self):
        r"""units.Quantity: Apparent elevation of the sun."""
        return parse_quantity(
            self.solar_position["apparent_elevation"].iloc[0], 'degrees')

    @property
    def azimuth(self):
        r"""units.Quantity: Azimuth angle of the sun."""
        return parse_quantity(
            self.solar_position["azimuth"].iloc[0], 'degrees')

    def relative_direction(self, up, north):
        r"""Get the direction from the scene to the sun.

        Args:
            up (np.ndarray): Unit vector for up in the scene.
            north (np.ndarray): Unit vector for north in the scene.

        Returns:
            np.ndarray: Unit vector from scene to sun.

        """
        from hothouse.blaster import SunRayBlaster
        blaster = SunRayBlaster(
            latitude=self.latitude.value, longitude=self.longitude.value,
            date=self.time, solar_altitude=self.apparent_elevation,
            solar_azimuth=self.azimuth,
            zenith=up.astype('f4'), north=north.astype('f4'),
            ground=np.zeros((3,), 'f4'),
            # direct_ppfd=self.ppfd_direct,
            # diffuse_ppfd=self.ppfd_diffuse,
        )
        return -blaster.forward

    # @property
    # def airmass(self):
    #     r"""pandas.DataFrame: Relative and absolute air mass."""
    #     return self.location.get_airmass(
    #         solar_position=self.solar_position,
    #         method=self.method_airmass,
    #     )

    @property
    def relative_airmass(self):
        r"""float: Relative (not pressure-adjusted) airmass at sea
        level."""
        return self.pvlib.atmosphere.get_relative_airmass(
            self.solar_position['apparent_zenith'])

    @property
    def absolute_airmass(self):
        r"""float: Absolute (pressure-adjusted) airmass."""
        return self.pvlib.atmosphere.get_absolute_airmass(
            self.relative_airmass, self.pressure)

    @property
    def linke_turbidity(self):
        r"""float: Linke Turibidity for the time/location."""
        return self.pvlib.clearsky.lookup_linke_turbidity(
            self.time_pv, self.latitude.value, self.longitude.value)

    @property
    def dni_extra(self):
        r"""units.Quantity: Extraterrestrial radiation incident on a
        surface normal to the sun (in W/m**2)."""
        return parse_quantity(
            self.pvlib.irradiance.get_extra_radiation(self.time_pv),
            "W m-2")

    @property
    def irradiance(self):
        r"""pandas.DataFrame: Solar irradiance."""
        if self._irradiance is None:
            self._irradiance = self.location.get_clearsky(
                self.time, model=self.method_irradiance,
                solar_position=self.solar_position,
                dni_extra=self.dni_extra,
                linke_turbidity=self.linke_turbidity,
                airmass_absolute=self.absolute_airmass,
            )
        return self._irradiance

    @property
    def dni(self):
        r"""units.Quantity: Direct normal irradiance"""
        return self.irradiance['dni']

    @property
    def dhi(self):
        r"""units.Quantity: Diffuse horizontal irradiance"""
        return self.irradiance['dhi']

    @property
    def ghi(self):
        r"""units.Quantity: Global horizontal irradiance"""
        return self.irradiance['ghi']

    @property
    def ppfd_direct(self):
        r"""units.Quantity: Direct photosynthetic photon flux density"""
        return self.eta_par * self.eta_photon * self.dni

    @property
    def ppfd_diffuse(self):
        r"""units.Quantity: Diffuse photosynthetic photon flux density"""
        return self.eta_par * self.eta_photon * self.dhi


class UnitSet(object):
    r"""Container for a unit set.

    Args:
        time (str, yggdrasil.units.Unit, optional): Time unit.
        length (str, yggdrasil.units.Unit, optional): Length unit.
        mass (str, yggdrasil.units.Unit, optional): Mass unit.
        angle (str, yggdrasil.units.Unit, optional): Angle unit.
        area (str, yggdrasil.units.Unit, optional): Area unit. If not
            provided, this unit will be calculated from the other
            provided units.
        density (str, yggdrasil.units.Unit, optional): Density unit. If
            not provided, this unit will be calculated from the other
            provided units.

    """

    _dimensions = [
        'time', 'length', 'mass', 'angle',
    ]
    _calculated_dimensions = [
        'area', 'density',
    ]

    def __init__(self, **kwargs):
        for k in self._dimensions + self._calculated_dimensions:
            v = kwargs.pop(k, None)
            if isinstance(v, str):
                v = units.Units(v)
            if k in self._calculated_dimensions:
                setattr(self, f'_{k}', v)
            else:
                setattr(self, k, v)
        assert not kwargs

    @property
    def dimensions(self):
        return [k for k in self._dimensions + self._calculated_dimensions
                if getattr(self, k) is not None]

    @property
    def area(self):
        if self._area is not None:
            return self._area
        if self.length is None:
            return None
        return self.length * self.length

    @property
    def density(self):
        if self._density is not None:
            return self._density
        if self.length is None or self.mass is None:
            return None
        return self.mass / self.area

    @property
    def units(self):
        out = units.Units()
        for k in self._dimensions:
            v = getattr(self, k)
            if v is not None:
                out = out * v
                # out *= v
        return out

    @classmethod
    def add_unit_arguments(cls, dest, prefix, help_template=None,
                           defaults=None):
        for k in cls._dimensions:
            iargs = (f'--{prefix}-units-{k}', )
            ikws = {'type': parse_units}
            if help_template:
                plural = k + 'es' if k.endswith('s') else k + 's'
                ikws['help'] = help_template.format(plural=plural)
            if defaults and k in defaults:
                ikws['default'] = defaults[k]
            dest._arguments.append((iargs, ikws))

    @classmethod
    def from_kwargs(cls, kwargs, prefix='', suffix='', pop=False):
        r"""Create the class by extracting units from the provided
        kwargs.

        Args:
            kwargs (dict): Dictionary that should be searched for units.
            prefix (str, optional): Prefix that should be added to units
                when search kwargs.
            suffix (str, optional): Suffix that should be added to units
                when search kwargs.
            pop (bool, optional): If True, units should be removed from
                kwargs if they are used to create the class.

        Returns:
            UnitSet: New unit set.

        """
        kws = {}
        if pop:
            kws = {
                k: kwargs.pop(f'{prefix}{k}{suffix}')
                for k in cls._dimensions + cls._calculated_dimensions
                if f'{prefix}{k}{suffix}' in kwargs
            }
        else:
            kws = {
                k: kwargs[f'{prefix}{k}{suffix}']
                for k in cls._dimensions + cls._calculated_dimensions
                if f'{prefix}{k}{suffix}' in kwargs
            }
        return cls(**kws)

    @classmethod
    def from_attr(cls, inst, prefix='', suffix=''):
        r"""Create the class by extracting units from attributes of the
        provided instance.

        Args:
            inst (object): Instance whose attributes should be checked
                for units.
            prefix (str, optional): Prefix that should be added to units
                when search kwargs.
            suffix (str, optional): Suffix that should be added to units
                when search kwargs.

        Returns:
            UnitSet: New unit set.

        """
        kws = {
            k: getattr(inst, f'{prefix}{k}{suffix}')
            for k in cls._dimensions + cls._calculated_dimensions
            if hasattr(inst, f'{prefix}{k}{suffix}')
        }
        return cls(**kws)

    def as_dict(self, prefix='', suffix='', include_missing=False,
                as_strings=False):
        r"""Get a dictionary of units in the set.

        Args:
            prefix (str, optional): Prefix that should be added to units
                in the returned dictionary.
            suffix (str, optional): Suffix that should be added to units
                in the returned dictionary.
            include_missing (bool, optional): If True, include missing
                units in the returned dictionary as None.
            as_strings (bool, optional): If True, units in the output
                dictionary should be strings.

        Returns:
            dict: Units dictionary.

        """
        out = {}
        for k in self._dimensions + self._calculated_dimensions:
            v = getattr(self, k)
            if v is not None or include_missing:
                out[f'{prefix}{k}{suffix}'] = str(v) if as_strings else v
        return out

    def add_units(self, x, dimension):
        r"""Add units to a quantity with the specified dimensionality.

        Args:
            x (units.QuantityArray): Quantity with units.
            dimension (str): Dimensions of units that should be added.

        Returns:
            units.QuantityArray: x converted to this unit system.

        """
        x_units = getattr(self, dimension)
        if not isinstance(x, units.QuantityArray):
            if isinstance(x, np.ndarray):
                x = units.QuantityArray(x, x_units)
            else:
                x = units.Quantity(x, x_units)
            return x
        assert x.is_compatible(x_units)
        return self.convert(x)

    def convert(self, x, strip=False):
        r"""Convert a quantity to this unit system.

        Args:
            x (units.QuantityArray): Quantity with units.
            strip (bool, optional): If True, return a float version with
                the units stripped.

        Returns:
            units.QuantityArray: x converted to this unit system.

        """
        if not isinstance(x, units.QuantityArray):
            return x
        if x.units.is_dimensionless():
            out = x
        else:
            out = x.to_system(self.units)
        if strip:
            out = float(out)
        return out


def quantity2timedelta(x):
    r"""Convert a quantity to a datetime timedelta difference.

    Args:
        x (units.Quantity): Quantity with time units.

    Returns:
        datetime.timedelta: Time difference.

    """
    order = ['days', 'seconds', 'microseconds']
    kws = {}
    for k in order:
        x = x.to(k)
        xk = int(x)
        x = x - units.Quantity(xk, k)
        kws[k] = xk
    return timedelta(**kws)


def timedelta2quantity(x):
    r"""Convert a datetime.timedelta instance into a quantity.

    Args:
        x (datetime.timedelta): Time difference.

    Returns:
        units.Quantity: Quantity with time units.

    """
    return units.Quantity(x.total_seconds(), 'seconds').to('days')


def get_mesh_dict(x):
    r"""Convert a ObjDict/PlyDict instance to a dictionary.

    Args:
        x (ObjDict, PlyDict): Mesh to convert.

    Returns:
        dict: Mapping between face/vertices and arrays.

    """
    out = x.as_array_dict()
    out.setdefault('face', np.empty((0,), dtype=np.int64))
    out.setdefault('vertex', np.empty((0, 3), dtype=np.double))
    out.setdefault('vertex_colors', np.empty((0, 3), dtype=np.int32))
    if isinstance(x, ObjDict):
        out['face'] -= 1
    return out


############################################################
# Enhanced dictionaries
############################################################

class ImmutableDictException(BaseException):
    pass


class DictWrapper(MutableMapping):
    r"""Abstract base class for dictionary wrappers."""

    def __init__(self, logger=None, logger_prefix=''):
        self.logger = logger
        self._logger_prefix = logger_prefix
        self._original_keys = set(self.keys(raw=True))
        # TODO: Remove this check
        # allkeys = list(self.keys())
        # if len(allkeys) != len(set(allkeys)):
        #     pdb.set_trace()
        # assert len(allkeys) == len(set(allkeys))

    @property
    @abstractmethod
    def storage(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        raise ImmutableDictException("Immutable")

    @property
    def mutable(self):
        r"""bool: True if keys can be added to the dictionary."""
        try:
            self.storage
            return True
        except ImmutableDictException:
            return False

    @property
    def flattened(self):
        r"""dict: Flattened version of the members."""
        return dict(self.items())

    @property
    def raw_flattened(self):
        r"""dict: Copy of raw destination dictionary (with prefix)"""
        return dict(self.items(raw=True))

    def __repr__(self):
        return self.flattened.__repr__()

    def __len__(self):
        return len(list(self._get_iterator()))

    def __iter__(self):
        for k, v in self._get_iterator():
            yield k

    def keys(self, raw=False, prefix=None):
        r"""Wrapped dictionary keys with prefixes removed.

        Args:
            raw (bool, optional): If True, the raw dictionary keys will
                be returned without the prefix removed.
            prefix (str, optional): If True, only return keys that start
                with this prefix.

        Returns:
            dict_keys: Keys view.

        """
        if not (raw or prefix):
            return super(DictWrapper, self).keys()
        return [k for k, v in self._get_iterator(raw=raw)
                if (prefix is None or k.startswith(prefix))]

    def items(self, raw=False):
        r"""Wrapped dictionary items with prefixes removed from keys.

        Args:
            raw (bool, optional): If True, the raw dictionary items will
                be returned without the prefix removed.

        Returns:
            dict_items: Items view.

        """
        if not raw:
            return super(DictWrapper, self).items()
        return [x for x in self._get_iterator(raw=raw)]

    def setdefault(self, k, v):
        if k in self:
            return
        self[k] = v

    @property
    def added(self):
        r"""set: Keys added to the dictionary."""
        return set(self.keys()) - self._original_keys

    @property
    def removed(self):
        r"""set: Keys removed from the dictionary."""
        return self._original_keys - set(self.keys())

    @classmethod
    def select_keys(cls, x, prefix=''):
        return {
            k: copy.deepcopy(v) for k, v in x.items()
            if (not prefix) or k.startswith(prefix)
        }

    @classmethod
    def remove_prefixed_keys(cls, x, prefix):
        for k in list(x.keys()):
            if k.startswith(prefix):
                del x[k]

    def _forward_key(self, k):
        return k

    def _reverse_key(self, k):
        return k

    def count_prefix(self, prefix, raw=False):
        r"""Count the number of keys that start with a prefix.

        Args:
            prefix (str): Prefix to count.
            raw (bool, optional): If True, the raw dictionary keys will
                be checked for the provided prefix.

        """
        return len(list(self.keys(raw=raw, prefix=prefix)))

    def copy_src2dst(self, kdst, ksrc=NoDefault, overwrite=False):
        r"""Copy a key/value pair from the source dictionary to the
        destination dictionary if it is present in the source dictionary
        and not present in the destination dictionary.

        Args:
            kdst (str): Key that should be assigned to in the destination.
            ksrc (str, optional): Key that should be copyied from the
                source dictionary. If not provided, kdst is used.
            overwrite (bool, optional): If True, overwrite any existing
                value in the destination dictionary.

        """
        if ksrc == NoDefault:
            ksrc = kdst
        val = NoDefault
        if ksrc in self:
            val = self[ksrc]
        elif kdst in self:
            val = self[kdst]
        kdst_raw = self._forward_key(kdst)
        if (((overwrite or kdst_raw not in self.storage)
             and val is not NoDefault)):
            self.storage[kdst_raw] = val

    def remove_cond(self, fcond):
        r"""Remove key/value pairs from this dictionary based on the
        value of a provided function.

        Args:
            fcond (callable): Function that takes a single raw key/value
                pair and returns True when the pair should be removed.

        """
        for k, v in list(self.items(raw=True)):
            if not fcond(k, v):
                del self[self._reverse_key(k)]

    def select_cond(self, fcond, transform=False, deepcopy=False):
        r"""Create a new PrefixedDict instance that only includes
        key/value pairs selected by the provided function.

        Args:
            fcond (callable): Selection function that takes a single
                key/value pair and returns True or False to indicate
                if the pair is selected.
            transform (bool, callable, optional): A function that should
                be used to transform raw keys for inclusion in the
                returned dictionary. If True, self._reverse_key will be
                used.
            deepcopy (bool, optional): If True, deepcopy values.

        Returns:
            dict: Dictionary containing selected members.

        """
        if transform is True:
            transform = self._reverse_key
        wrapped = {}
        for k, v in self.items(raw=True):
            if fcond(k, v):
                kx = transform(k) if transform else k
                if deepcopy:
                    wrapped[kx] = copy.deepcopy(v)
                else:
                    wrapped[kx] = v
        return wrapped

    def remove_prefix(self, prefix, **kwargs):
        r"""Remove keys that start with the provided prefix.

        Args:
            prefix (str): Prefix to remove keys with.
            **kwargs: Additional keyword arguments are passed to
                remove_cond.

        """

        def fcond(k, v):
            return ((not prefix) or k.startswith(prefix))

        self.remove_cond(fcond, **kwargs)

    def select_prefix(self, prefix, strip=False, **kwargs):
        r"""Create a new PrefixedDict instance that only includes keys
        that start with a given prefix.

        Args:
            prefix (str): Prefix to filter on.
            strip (bool, optional): If True, the prefix should be
                stripped in the returned dictionary.
            **kwargs: Additional keyword arguments are passed to
                select_cond.

        Returns:
            dict: Dictionary containing selected members.

        """

        def fcond(k, v):
            return ((not prefix) or k.startswith(prefix))

        def fstrip(k):
            if not prefix:
                return k
            return k.split(prefix, 1)[-1]

        if strip:
            assert 'transform' not in kwargs
            kwargs['transform'] = fstrip
        return self.select_cond(fcond, **kwargs)

    def strip_prefix(self, prefix, keys=None):
        r"""Remove a prefix from keys in the dictionary.

        Args:
            prefix (str): Prefix to strip.
            keys (list, optional): Set of keys to remove the prefix from.
                If not provided, all keys will be used and an error will
                be raised if any keys do not start with the prefix.

        """
        if not prefix:
            return
        if keys is None:
            keys = list(self.keys())
        for k in keys:
            if not k.startswith(prefix):
                raise KeyError(f"Key \"{k}\" does not "
                               f"start with prefix \"{prefix}\"")
            ky = k.split(prefix, 1)[-1]
            self[ky] = self.pop(k)

    def update_missing(self, other):
        r"""Update the dictionary only will values that are not present.

        Args:
            other (dict): Values to update the dictionary with.

        """
        for k, v in other.items():
            self.setdefault(k, v)

    @classmethod
    def coerce(cls, x, **kwargs):
        r"""Coerce an object to be a DictWrapper compatible instance.

        Args:
            x (dict, DictWrapper, tuple, list): Object that can be
                interpreted as a DictWrapper instance.
            **kwargs: Additional keyword arguments are passed to the
                constructor for the interpreted instance.

        Returns:
            DictWrapper: Coerced instance.

        """
        if isinstance(x, tuple):
            assert len(x) == 2
            return PrefixedDict(x, **kwargs)
        elif isinstance(x, list):
            x = DictSet(x)
        if isinstance(x, (dict, DictWrapper)):
            if kwargs:
                return PrefixedDict(x, **kwargs)
            return x
        raise TypeError(type(x))

    @abstractmethod
    def _get_iterator(self, raw=False, prev=None):
        r"""Yields member key/item pairs.

        Args:
            raw (bool, optional): If True, raw keys should be yielded.
            prev (list, optional): Set of keys already yielded.

        Yields:
            tuple: Member items.

        """
        raise NotImplementedError

    @contextlib.contextmanager
    def temporary_prefix(self, prefix, append=False, report_change=False):
        r"""Temporarily set a prefix for the dictionary with the context.
        When the context exits, any previous prefix will be restored and
        any added parameters will be maintained.

        Args:
            prefix (str): Prefix to use within the context.
            append (bool, optional): If True, the prefix should be
                appended to the end of the current prefix.
            report_change (str, bool, optional): If True, changes
                that occur in the context will be reported. If a string
                is provided, it will be used as a prefix for the log
                message.

        Yields:
            DictWrapper: View of the dictionary with the provided prefix.

        """
        assert not append
        if report_change is True:
            report_change = ''
        if isinstance(report_change, str):
            cm = self.report_change(prefix=report_change)
        else:
            cm = contextlib.nullcontext()
        with cm:
            yield PrefixedDict(self, prefix=prefix)

    @classmethod
    def assert_keys_match(cls, x, keys0, logger=None):
        if not list(x.keys()) == keys0:
            message = (
                f'ADDED:   {set(x.keys()) - set(keys0)}\n'
                f'REMOVED: {set(keys0) - set(x.keys())}'
            )
            if logger:
                logger.log(message, force=True)
            else:
                print(message)
            pdb.set_trace()
        assert list(x.keys()) == keys0

    @property
    def logger_prefix(self):
        r"""str: Prefix for log messages."""
        return self._logger_prefix

    def report_diff(self, keys0, keys1, prefix='', suffix=''):
        r"""Report on a difference between two key sets.

        Args:
            keys0 (set): Set of keys from a prior state.
            keys1 (set): Set of keys from a state after keys0.
            prefix (str, optional): Prefix for report message.
            suffix (str, optional): Suffix for report message.

        """
        added = keys1 - keys0
        removed = keys0 - keys1
        msg = ''
        if added:
            msg += f'ADDED {added}.'
        if removed:
            msg += ' ' if added else ''
            msg += f'REMOVED {removed}.'
        if not (added or removed):
            msg += 'NO CHANGE'
            return
        self.logger.log(f'{self.logger_prefix}[{self.prefix}]: '
                        f'{prefix}{msg}{suffix}')

    @contextlib.contextmanager
    def report_change(self, prefix='', suffix='',
                      assert_no_change=False, **kwargs):
        r"""Report on how the dictionary changed within the context.

        Args:
            prefix (str, optional): Prefix for report message.
            suffix (str, optional): Suffix for report message.
            assert_no_change (bool, optional): If True, assert that the
                keys did not change.
            **kwargs: Additional keyword arguments are passed to the
                keys method to get the set of keys before & after the
                context.

        """
        kwargs.setdefault('raw', True)
        keys0 = set(self.keys(**kwargs))
        try:
            yield
        finally:
            keys1 = set(self.keys(**kwargs))
            if assert_no_change:
                assert keys0 == keys1
            self.report_diff(keys0, keys1, prefix=prefix, suffix=suffix)


pprint.PrettyPrinter._dispatch[DictWrapper.__repr__] = (
    pprint.PrettyPrinter._pprint_dict
)


class SimpleWrapper(DictWrapper):
    r"""Dictionary wrapper that only allows certain keys.

    Args:
        wrapped (dict, optional): Wrapped dictionary.
        immutable (bool, optional): If True, the dictionary should not
            be modified.
        ordered (bool, optional): If True and wrapped not provided, an
            OrderedDict will be used.
        **kwargs: Additional keyword arguments are passed to the parent
            constructor.

    """

    def __init__(self, wrapped=None, immutable=False, ordered=False,
                 **kwargs):
        if wrapped is None:
            if ordered:
                wrapped = OrderedDict()
            else:
                wrapped = {}
        self._wrapped = wrapped
        self._immutable = immutable
        super(SimpleWrapper, self).__init__(**kwargs)

    @property
    def storage(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        if self._immutable:
            raise ImmutableDictException("Immutable")
        return self._wrapped

    def move_to_end(self, key, last=True):
        r"""Move a key to the end of a sorted ordered dictionary.

        Args:
            key (str): Key to move.
            last (bool, optional): If True, move the key to the beginning
                instead of the end.

        """
        if not isinstance(self._wrapped, OrderedDict):
            raise NotImplementedError
        self.storage.move_to_end(key, last=last)

    def _get_iterator(self, raw=False, prev=None):
        r"""Yields member key/item pairs.

        Args:
            raw (bool, optional): If True, raw keys should be yielded.
            prev (list, optional): Set of keys already yielded.

        Yields:
            tuple: Member items.

        """
        if prev is None:
            prev = []
        for k, v in self._wrapped.items():
            try:
                kx = self._reverse_key(k)
            except KeyError:
                continue
            if kx in prev:
                continue
            if raw:
                yield (k, v)
            else:
                yield (kx, v)
            prev.append(kx)

    def __getitem__(self, k):
        return self._wrapped[self._forward_key(k)]

    def __setitem__(self, k, v):
        self.storage[self._forward_key(k)] = v

    def __delitem__(self, k):
        del self.storage[self._forward_key(k)]
        assert k not in self

    # Methods for nested DictSet
    @property
    def members(self):
        r"""list: Members of the wrapped dictionary. Only valid if
        the wrapped dictionary is a DictSet."""
        return self._wrapped.members

    @members.setter
    def members(self, x):
        self._wrapped.members = x

    def insert(self, idx, member, add_prefix=False, **kwargs):
        r"""Add a new member to the set.

        Args:
            idx (index): Index to insert the new member add.
            member (dict): New member.
            add_prefix (str, bool, optional): Prefix that should be added
                to the keys in x. If True, the current prefix should be
                added.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        if add_prefix is True:
            add_prefix = self.prefix
        return self._wrapped.insert(idx, member, add_prefix=add_prefix,
                                    **kwargs)

    def append(self, member, add_prefix=False, **kwargs):
        r"""Add a new member to the end of the set.

        Args:
            member (dict): New member.
            add_prefix (str, bool, optional): Prefix that should be added
                to the keys in x. If True, the current prefix should be
                added.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        if add_prefix is True:
            add_prefix = self.prefix
        return self._wrapped.append(member, add_prefix=add_prefix,
                                    **kwargs)


class PrefixedDict(SimpleWrapper):
    r"""Dictionary wrapper that allows for adding a prefix to keys when
    accessing the wrapped dictionary.

    Args:
        wrapped (dict, optional): Wrapped dictionary.
        prefix (str, optional): Prefix to add to keys.
        **kwargs: Additional keyword arguments are passed to the parent
            constructor.

    """

    def __init__(self, wrapped=None, prefix=None, **kwargs):
        if wrapped is None:
            wrapped = {}
        if isinstance(wrapped, tuple):
            if prefix is None:
                prefix = wrapped[0]
                wrapped = wrapped[1]
            else:
                wrapped = PrefixedDict(wrapped[1], prefix=wrapped[0])
        if prefix is None:
            prefix = ''
        self.prefix = prefix
        super(PrefixedDict, self).__init__(wrapped=wrapped, **kwargs)

    @property
    def logger_prefix(self):
        r"""str: Prefix for log messages."""
        return f'{self._logger_prefix}[{self.prefix}]'

    def _forward_key(self, k):
        if not self.prefix:
            return k
        return f'{self.prefix}{k}'

    def _reverse_key(self, k):
        if not self.prefix:
            return k
        if not k.startswith(self.prefix):
            raise KeyError(f'Key \"{k}\" does not start with prefix '
                           f'\"{self.prefix}\"')
        try:
            assert k.startswith(self.prefix)
        except AssertionError:
            print(k, self.prefix)
            pdb.set_trace()
            raise
        return k.split(self.prefix, 1)[-1]

    def select_prefix(self, prefix=None, strip=False, **kwargs):
        r"""Create a new PrefixedDict instance that only includes keys
        that start with a given prefix.

        Args:
            prefix (str, optional): Prefix to filter on. If not provided,
                the prefix for the target will be used.
            strip (bool, optional): If True, the prefix should be
                stripped in the returned dictionary.
            **kwargs: Additional keyword arguments are passed to the
                parent method.

        Returns:
            PrefixedDict: Dictionary containing selected members.

        """
        if prefix is None:
            prefix = self.prefix
        out = super(PrefixedDict, self).select_prefix(prefix, **kwargs)
        out_prefix = prefix if not kwargs.get('strip', False) else ''
        return PrefixedDict(out, prefix=out_prefix)

    @contextlib.contextmanager
    def temporary_prefix(self, prefix, append=False, report_change=False):
        r"""Temporarily set a prefix for the dictionary with the context.
        When the context exits, any previous prefix will be restored and
        any added parameters will be maintained.

        Args:
            prefix (str): Prefix to use within the context.
            append (bool, optional): If True, the prefix should be
                appended to the end of the current prefix.
            report_change (str, bool, optional): If True, changes
                that occur in the context will be reported. If a string
                is provided, it will be used as a prefix for the log
                message.

        Yields:
            DictWrapper: View of the dictionary with the provided prefix.

        """
        if append:
            prefix = self.prefix + prefix
        if report_change is True:
            report_change = ''
        if isinstance(report_change, str):
            cm = self.report_change(prefix=report_change)
        else:
            cm = contextlib.nullcontext()
        old_prefix = self.prefix
        with cm:
            self.prefix = prefix
            try:
                yield self
            finally:
                self.prefix = old_prefix


@contextlib.contextmanager
def temporary_prefix(x, prefix, **kwargs):
    r"""Create a wrapper around a dictionary with a temporary prefix that
    should be added when accessing dictionary elements.

    Args:
        x (dict, DictWrapper): Dictionary to wrap with a prefix.
        prefix (str): Prefix to use within the context.
        **kwargs: Additional keyword arguments are passed to the
            temporary_prefix PrefixedDict method.

    Yields:
        PrefixedDict: Dictionary with prefix wrapper.

    """
    if not isinstance(x, DictWrapper):
        x = PrefixedDict(x)
    with x.temporary_prefix(prefix, **kwargs):
        yield x


class DictSet(DictWrapper):
    r"""Container for chaining multiple dictionaries.

    Args:
        members (list): Set of dictionaries that should be accessed in
            the order provided.
        **kwargs: Additional keyword arguments are passed to the parent
            constructor.

    """

    def __init__(self, members, **kwargs):
        assert isinstance(members, list)
        self.members = []
        for x in members:
            self.append(x)
        super(DictSet, self).__init__(**kwargs)

    @classmethod
    def coerce_member(cls, x, add_prefix=False, **kwargs):
        r"""Coerce an object to a DictWrapper instance.

        Args:
            x (object): Dictionary-like object.
            add_prefix (str, optional): Prefix that should be added to the
                keys in x.
            **kwargs: Additional keyword arguments are passed to
                DictWrapper.coerce.

        """
        if add_prefix:
            assert isinstance(x, (dict, DictWrapper))
            x = {add_prefix + k: v for k, v in x.items()}
        if isinstance(x, DictWrapper) and not kwargs:
            return x
        kwargs.setdefault('immutable', True)
        return DictWrapper.coerce(x, **kwargs)

    @property
    def storage(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        for x in self.members:
            if x.mutable:
                return x
        raise ImmutableDictException("No immutable members")

    def __getitem__(self, k):
        for x in self.members:
            if k in x:
                return x[k]
        raise KeyError(k)

    def __setitem__(self, k, v):
        storage = self.storage
        for x in self.members:
            if x is storage:
                storage[k] = v
                return
            if k in x:
                raise ImmutableDictException(
                    f'Setting \"{k}\" will not change the contents as '
                    f'it is present in an immutable member proceeding '
                    f'the first mutable member'
                )

    def __delitem__(self, k):
        for x in self.members:
            if k not in x:
                continue
            if not x.mutable:
                raise ImmutableDictException(
                    f'Cannot delete \"{k}\". It is present in an '
                    f'immutable member.'
                )
            del x[k]

    def __iter__(self):
        prev = []
        for x in self.members:
            for k, v in x.items():
                if k in prev:
                    continue
                yield k
                prev.append(k)

    def _get_iterator(self, raw=False, prev=None):
        if prev is None:
            prev = []
        for x in self.members:
            for k, v in x._get_iterator(raw=raw, prev=prev):
                yield (k, v)

    @contextlib.contextmanager
    def temporary_prefix(self, prefix, per_member=False, **kwargs):
        r"""Temporarily set a prefix for the dictionary with the context.
        When the context exits, any previous prefix will be restored and
        any added parameters will be maintained.

        Args:
            prefix (str): Prefix to use within the context.
            per_member (bool, optional): If True, add prefixes on a
                member-by-member basis.
            **kwargs: Additional keyword arguments are passed to
                temporary_prefix for each member if per_member is True,
                and the parent method otherwise.

        Yields:
            DictWrapper: View of the dictionary with the provided prefix.

        """
        if not per_member:
            with super(DictSet, self).temporary_prefix(
                    prefix, **kwargs) as child:
                yield child
            return
        with contextlib.ExitStack() as stack:
            for x in self.members:
                stack.enter_context(x.temporary_prefix(prefix, **kwargs))
            yield self

    def insert(self, idx, member, **kwargs):
        r"""Add a new member to the set.

        Args:
            idx (index): Index to insert the new member add.
            member (dict): New member.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        self.members.insert(idx, self.coerce_member(member, **kwargs))

    def append(self, member, **kwargs):
        r"""Add a new member to the end of the set.

        Args:
            member (dict): New member.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        self.members.append(self.coerce_member(member, **kwargs))


class DataProcessor:
    r"""Class for processing crop data into parameters.

    Args:
        crop (str): Crop name.
        metadata (dict, optional): Metadata for the crop.
        units (dict, optional): Unit system that data will be stored in.

    """

    _schema = {
        'type': 'object',
        'required': ['metadata', 'units', 'data'],
        'properties': {
            'metadata': {
                'type': 'object',
                'required': ['crop', 'year', 'authors'],
                'properties': {
                    'crop': {'type': 'string'},
                    'year': {'type': 'string'},
                    'authors': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'minItems': 1,
                    },
                    'sources': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'minItems': 1,
                    },
                },
            },
            'units': {
                'type': 'object',
                'required': ['length', 'mass', 'time', 'angle'],
                'additionalProperties': {
                    # TODO: Change to units once added to rapidjson
                    'type': 'string',
                },
            },
            'data': {
                'type': 'object',
                'additionalProperties': {
                    'type': 'object',
                    'additionalProperties': {
                        'type': 'object',
                        'additionalProperties': {
                            'type': '1darray',
                            'subtype': 'float',
                            'precision': 8,
                        },
                    },
                },
            },
        },
        'additionalProperties': False,
    }

    def __init__(self, crop=None, year=None, metadata=None, units=None):
        if metadata is None:
            metadata = {}
        metadata.setdefault('crop', crop)
        metadata.setdefault('year', year)
        metadata.setdefault('sources', [])
        if units is None:
            units = {
                'length': 'cm',
                'mass': 'kg',
                'time': 'days',
                'angle': 'degrees',
            }
        self.metadata = metadata
        self.units = units
        self.data = {}
        if crop is not None:
            assert self.crop == crop
        if year is not None:
            assert self.year == year

    @classmethod
    def from_file(cls, fname, **kwargs):
        r"""Create a DataProcessor instance from a file name.

        Args:
            fname (str): File containing parameter measurements in JSON.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            DataProcessor: Processory instance with data loaded.

        """
        out = cls(**kwargs)
        out.read(fname)
        return out

    @classmethod
    def ids_from_file(cls, fname, **kwargs):
        r"""Get available ids from a datafile.

        Args:
            fname (str): File containing parameter measurements in JSON.
            **kwargs: Additional keyword arguments are passed to
                from_file.

        Returns:
            list: ID strings.

        """
        try:
            return cls.from_file(fname, **kwargs).ids
        except ValueError:
            if fname is None and kwargs.get('crop', None):
                return cls.available_ids(kwargs.pop('crop'), **kwargs)
            raise

    @classmethod
    def base_id_from_file(cls, *args, **kwargs):
        r"""Get the base ID from a datafile.

        Args:
            *args, **kwargs: Additional arguments are passed to
                ids_from_file.

        Returns:
            str: Base id.

        """
        ids = cls.ids_from_file(*args, **kwargs)
        if not ids:
            return None
        return ids[0]

    @classmethod
    def output_name(cls, crop, year):
        r"""Create an output file name for a given crop and year.

        Args:
            crop (str): Crop name.
            year (str): Year that data was collected in.

        Returns:
            str: Data file name.

        """
        if not (crop and year):
            raise ValueError("Both crop and year must be provided to "
                             "generate a file name")
        return os.path.join(cfg['directories']['input'],
                            f'{crop}_{year}.json')

    @classmethod
    def available_files(cls, crop=None, id=None, year=None):
        r"""Locate files containing data for a certain crop and/or year.

        Args:
            crop (str, optional): Crop name.
            id (str, optional): ID string to find files for.
            year (str, optional): Year that data was collected in.

        Returns:
            list: Matching files.

        """
        if isinstance(crop, list):
            out = []
            for x in crop:
                out += cls.available_files(crop=x, id=id, year=year)
            return out
        if isinstance(year, list):
            out = []
            for x in year:
                out += cls.available_files(crop=crop, id=id, year=x)
            return out
        if crop is None:
            crop = '*'
        if year is None:
            year = '*'
        regex = cls.output_name(crop, year)
        out = sorted(glob.glob(regex))
        if id is not None:
            if not isinstance(id, list):
                id = [id]
            out_id = []
            for fname in out:
                x = cls.from_file(fname)
                if any(iid in x.ids for iid in id):
                    out_id.append(fname)
            return out_id
        return out

    @classmethod
    def available_years(cls, crop, **kwargs):
        r"""Determine the years in which there is data available for a
        given crop.

        Args:
            crop (str): Crop name.
            **kwargs: Additional keyword arguments are passed to
                available_files.

        Returns:
            list: Available years.

        """
        out = set()
        for fname in cls.available_files(crop, **kwargs):
            x = cls.from_file(fname)
            out |= set([x.year])
        if kwargs.get('year', None) is not None:
            years = kwargs['year']
            if not isinstance(years, list):
                years = [years]
            out &= set(years)
        return sorted(list(out))

    @classmethod
    def available_ids(cls, crop, **kwargs):
        r"""Determine the IDs for which there is data available for a
        given crop.

        Args:
            crop (str): Crop name.
            **kwargs: Additional keyword arguments are passed to
                available_files.

        Returns:
            list: Available ids.

        """
        out = set()
        for x in cls.available_files(crop, **kwargs):
            out |= set(cls.from_file(x).ids)
        if kwargs.get('id', None) is not None:
            ids = kwargs['id']
            if not isinstance(ids, list):
                ids = [ids]
            out &= set(ids)
        return sorted(list(out))

    @classmethod
    def available_param(cls, names, **kwargs):
        r"""Get a list of available data set parameters.

        Args:
            names (list): Names of data parameters to include.
            **kwargs: Additional keyword arguments are passed to
                available_files and used to select a subset of the
                available data sets.

        Returns:
            list: Set of tuples containing available named parameters.

        """
        out = []
        for fname in cls.available_files(**kwargs):
            x = cls.from_file(fname)
            xparam = [getattr(x, k) if k != 'id' else None for k in names]
            if 'id' in names:
                ids = kwargs.get('id', None)
                if isinstance(ids, str):
                    ids = [ids]
                idx = names.index('id')
                xids = (
                    x.ids if not ids
                    else [iid for iid in ids if iid in x.ids]
                )
                for iid in xids:
                    xparam[idx] = iid
                    out.append(tuple(xparam))
            else:
                out.append(tuple(xparam))
        return out

    @cached_property
    def param(self):
        r"""dict: Parameters saved to file."""
        return {
            'metadata': self.metadata,
            'units': self.units,
            'data': self.data,
        }

    @property
    def crop(self):
        r"""str: Crop that data pertains to."""
        return self.metadata['crop']

    @property
    def year(self):
        r"""str: Year that the data was collected."""
        return self.metadata['year']

    @property
    def ids(self):
        r"""list: Set of IDs for genotypes that have data."""
        return sorted(list(self.data))

    def parameter_names(self, idstr=None):
        r"""Get the set of parameters present in the data for the
        provided ID string.

        Args:
            idstr (str, optional): ID string for the genotype. If not
                provided all of the parameters present in any of the
                genotype data will be used.

        Returns:
            list: Parameter names.

        """
        if idstr is None:
            out = set()
            for idstr in self.ids:
                out |= set(self.parameter_names(idstr=idstr))
            return list(out)
        return list(self.data[idstr].keys())

    def process_csv(self, fname, genotype=None, **kwargs):
        r"""Process data from a file.

        Args:
            fname (str): Path to CSV file that data should be loaded
                from.
            genotype (str, optional): Name of the genotype that the data
                in fname pertain to or that should be processed. Required
                if genotype is not specified in the data.
            **kwargs: Additional keyword arguments are passed to
                extract_parameter_data.

        """
        if not (os.path.isfile(fname) or os.path.isabs(fname)):
            fname = os.path.join(cfg['directories']['input'], fname)
        print(f"Loading data from \"{fname}\"")
        df0 = pd.read_csv(fname)
        if kwargs.get('debug', False):
            print(df0)
        genotypic_classes = sorted(list(set(df0['Class'])))
        if genotype is None:
            if "Genotype" not in df0:
                raise RuntimeError(f'Genotype not provided and not '
                                   f'present in the file \"{fname}\"')
            genotypes = sorted(list(set(df0['Genotype'])))
        else:
            genotypes = [genotype]
        for genotype in genotypes:
            if 'Genotype' in df0:
                df_genotype = self.select_data(df0, genotype=genotype)
            else:
                df_genotype = df0
            genotypic_classes = sorted(list(set(df_genotype['Class'])))
            for genotypic_class in genotypic_classes:
                df_class = self.select_data(
                    df_genotype, genotypic_class=genotypic_class)
                if df_class.empty:
                    raise ValueError(f"No data found for name "
                                     f"\"{genotypic_class}\"")
                idstr = f'{genotype}_{genotypic_class}'
                self.data.setdefault(idstr, {})
                self.data[idstr].update(
                    **self.extract_parameter_data(df_class, **kwargs))
        self.metadata['sources'].append(os.path.basename(fname))

    @classmethod
    def extract_parameter_data(cls, df, regex=None, component=None,
                               parameter=None, noffset=-1, debug=False):
        r"""Extract data for crop class parameters.

        Args:
            df (pandas.DataFrame): Data frame to extract parameters from.
            regex (str, optional): Regular expression that should be used
                to identify columns and extract parameter names.
            component (str, optional): Component that the data pertains
                to.
            parameter (str, optional): Parameter that the data pertains
                to.
            noffset (int, optional): Offset that should be applied to the
                parsed n value.
            debug (bool, optional): If True, turn on debugging.

        Returns:
            dict: Extracted parameters.

        """
        if regex is None:
            regex = r'^(?P<stage>[VR])(?P<n>\d+) (?P<parameter>[a-zA-Z]+)'
        out = {}
        for col in df.filter(regex=regex):
            match = re.search(regex, col)
            if not match:
                raise ValueError(f"Failed to parse column name \"{col}\" "
                                 f"with regex \"{regex}\"")
            if parameter is not None:
                iparameter = parameter
            else:
                iparameter = match['parameter']
            if component is not None:
                icomponent = component
            elif 'component' in match.groupdict():
                icomponent = match['component']
            else:
                icomponent = ''
            param = f'{icomponent.title()}{iparameter.title()}'
            n = str(int(match['n'].replace(' ', '')) + noffset)
            nvalue = df[col].to_numpy().flatten().astype(np.float64)
            if np.isnan(nvalue).sum() == len(nvalue):
                continue
            # TODO: Add units?
            out.setdefault(param, {})
            if n in out[param]:
                raise ValueError(f"Duplicate n={n}?")
            assert n not in out[param]
            out[param][n] = nvalue
        if debug:
            pprint.pprint(out)
            pdb.set_trace()
        return out

    @classmethod
    def select_data(cls, df, genotype=None, genotypic_class=None,
                    parameter=None, n=None, regex=None):
        r"""Select a subset of observed data.

        Args:
            df (pandas.DataFrame): Data frame that should be filtered.
            genotype (str, optional): Genotype that should be selected.
            genotypic_class (str, optional): Genotypic class that should
                be selected
            parameter (str, optional): Parameter that should be selected.
            n (int, optional): Phytomer count that should be selected.
            regex (str, optional): Regex to use for data selection.

        Returns:
            pandas.DataFrame: Selected data.

        """
        if genotype is not None:
            df = df.loc[df['Genotype'] == genotype]
        if genotypic_class is not None:
            df = df.loc[df['Class'] == genotypic_class]
        if parameter is not None:
            df = df.filter(regex=f' {parameter.title()}$')
        if n is not None:
            assert not (n % 1)
            n = int(n) + 1
            df = df.filter(regex=f'^[VR]{n} ')
        if regex is not None:
            df = df.filter(regex=regex)
        return df

    def parametrize(self, idstr, args=None, generator=None,
                    default_profile='normal'):
        r"""Create crop parameters from the raw data collected for those
        parameters.

        Args:
            idstr (str): ID string for the genotype that should be
                parametrized.
            args (ParsedArguments, optional): Parsed arguments that can
                be used to specify profiles that should be parametrized
                for different fields.
            generator (PlantGenerator, optional): Crop generator that
                should be parametrized. If not provided, the generator for
                the crop specified by the data will be used.
            default_profile (str, optional): Default profile that should
                be used if one is not specified by args.

        Returns:
            dict: Parameters.

        """
        from canopy_factory.crops.base import DistributionPlantParameter
        if generator is None:
            generator = get_class_registry().get('crop', self.crop)
        out = {}
        component_nmax = {}
        scale_by_nmax = {}
        for k, v in self.data[idstr].items():
            try:
                kclass = generator.get_class(k)
            except KeyError:
                warnings.warn(f'No class for parameter \"{k}\"')
                continue
            is_upper = [x.isupper() for x in k]
            component = k[:(is_upper[1:].index(True) + 1)]
            kunits = (self.units[kclass._unit_dimension]
                      if kclass._unit_dimension else None)
            profile = getattr(args, f'{k}Dist', None)
            if profile is None:
                profile = default_profile
            if profile != 'normal':
                out[k] = 1.0
                out[f'{k}Dist'] = profile
            nvals = np.array([int(n) for n in v.keys()])
            nmax = int(max(nvals))
            component_nmax[component] = max(
                nmax, component_nmax.get(component, 0))
            data = np.vstack(list(v.values())).T
            # if kunits:
            #     data = units.QuantityArray(data, kunits)
            param_values = DistributionPlantParameter.parametrize_dist(
                data, profile=profile, axis=0,
            )
            if profile == 'normal':
                param_values['RelStdDev'] = param_values.pop(
                    'StdDev') / param_values['Mean']
            for kdist, v in param_values.items():
                if kunits is not None and kdist != 'RelStdDev':
                    v = units.QuantityArray(v, kunits)
                if profile != 'normal':
                    kdst = f'{k}Dist{kdist}'
                elif kdist == 'Mean':
                    kdst = k
                else:
                    kdst = f'{k}{kdist}'
                out[f'{kdst}'] = 1.0
                out[f'{kdst}NFunc'] = 'interp'
                out[f'{kdst}NFuncXVals'] = nvals  # / nmax
                out[f'{kdst}NFuncYVals'] = v
                scale_by_nmax.setdefault(component, [])
                scale_by_nmax[component].append(f'{kdst}NFuncXVals')
        for k, v in scale_by_nmax.items():
            for x in v:
                out[x] = out[x] / component_nmax[k]
        for k, v in component_nmax.items():
            out[f'{k}NMax'] = v
        return out

    def read(self, fname=None):
        r"""Read processed data from a JSON file.

        Args:
            fname (str, optional): Name of the file to read. If not
                provided, one will be generated using output_name.

        """
        # TODO: Use schema to validate the loaded data
        if fname is None:
            fname = self.output_name(self.crop, self.year)
        if not (os.path.isfile(fname) or os.path.isabs(fname)):
            fname = os.path.join(cfg['directories']['input'], fname)
        with open(fname, 'r') as fd:
            param = rapidjson.load(fd)
        rapidjson.validate(param, self._schema)
        for k, v in param.items():
            dst = getattr(self, k)
            dst.clear()
            dst.update(**v)

    def write(self, fname=None):
        r"""Write the processed data to a file as a JSON.

        Args:
            fname (str, optional): Name of the file to output. If not
                provided, one will be generated using output_name.

        """
        if not self.data:
            raise RuntimeError(f'Data is empty. {fname} will not be '
                               f'created.')
        if fname is None:
            fname = self.output_name(self.crop, self.year)
        if not os.path.isabs(fname):
            fname = os.path.join(cfg['directories']['input'], fname)
        rapidjson.validate(self.param, self._schema)
        with open(fname, 'w') as fd:
            rapidjson.dump(self.param, fd,
                           write_mode=rapidjson.WM_PRETTY)
        print(f"Saved data to \"{fname}\"")

    def plot_var(self, ax, idstr, var, nlimits=None):
        r"""Add data for a single variable to an axes.

        Args:
            ax (matplotlib.axes.Axes): Axes.
            idstr (str): ID string for the genotype that should be plot.
            var (str): Name of variable to plot.
            nlimits (list): Existing list that should be updated with
                limits on n.

        """
        if var not in self.data[idstr]:
            ax.errorbar([], [], [], label=f'{idstr} ({self.year})')
            return
        order = sorted(list(self.data[idstr][var].keys()), key=int)
        x = [int(k) + 1 for k in order]
        yavg = [np.nanmean(self.data[idstr][var][k]) for k in order]
        ystd = [np.nanstd(self.data[idstr][var][k]) for k in order]
        ax.errorbar(x, yavg, ystd, label=f'{idstr} ({self.year})')
        if nlimits is not None:
            nlimits[0] = min(nlimits[0], min(x))
            nlimits[1] = max(nlimits[1], max(x))

    def plot(self, fname=None, other=None, parameters=None):
        r"""Plot the dependence of parameters on phytomer number.

        Args:
            fname (str, optional): File where the plot should be saved.
            other (DataProcessor, optional): Other set of parameters to
                include.
            parameters (list, optional): Set of parameters to plot.

        """
        import matplotlib.pyplot as plt
        if parameters is None:
            parameters = sorted(self.parameter_names())
            if other is not None:
                parameters = sorted(list(
                    set(parameters) | set(other.parameter_names())))
        ncol = 1 if len(parameters) == 1 else 2
        nrow = int(np.ceil(len(parameters) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(18, 3 * nrow),
                                layout='constrained')
        nlimits = [3, 1]
        for ax, v in zip(axs.flat, parameters):
            ax.set_xlabel('Phytomer Number')
            ax.set_ylabel(v)  # TODO: Add units
            for idstr in self.ids:
                self.plot_var(ax, idstr, v, nlimits=nlimits)
            if other:
                for idstr in other.ids:
                    other.plot_var(ax, idstr, v, nlimits=nlimits)
        nlimits[0] -= 1
        nlimits[1] += 1
        for ax in axs.flat:
            ax.set_xlim(*nlimits)
        axs.flat[0].legend()
        if fname:
            if not (os.path.isfile(fname) or os.path.isabs(fname)):
                fname = os.path.join(cfg['directories']['input'], fname)
            print(f'Saving plot to \"{fname}\"')
            fig.savefig(fname)
        else:
            plt.show()
