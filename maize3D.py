import os
import sys
import pprint
import pdb
import copy
import argparse
import numpy as np
import pandas as pd
import scipy
import functools
import warnings
import subprocess
from collections import OrderedDict
from datetime import datetime, timedelta
from openalea.lpy import Lsystem
import openalea.plantgl.all as pgl
import openalea.plantgl.math as pglmath
from openalea.plantgl.math import Vector2, Vector3, Vector4
from openalea.plantgl.scenegraph import (
    NurbsCurve2D, NurbsCurve, NurbsPatch)
from openalea.plantgl.all import Tesselator
from yggdrasil import units, rapidjson
from yggdrasil.serialize.PlySerialize import PlyDict
from yggdrasil.serialize.ObjSerialize import ObjDict
from yggdrasil.communication.PlyFileComm import PlyFileComm
from yggdrasil.communication.ObjFileComm import ObjFileComm
from yggdrasil.communication.AsciiTableComm import AsciiTableComm


functools_cached_property = getattr(functools, "cached_property", None)
_default_task = 'generate'
_source_dir = os.path.abspath(os.path.dirname(__file__))
_param_dir = os.path.join(_source_dir, 'param')
_mesh_dir = os.path.join(_source_dir, 'meshes')
_traced_mesh_dir = os.path.join(_source_dir, 'traced_meshes')
_input_dir = os.path.join(_source_dir, 'input')
_image_dir = os.path.join(_source_dir, 'images')
_movie_dir = os.path.join(_source_dir, 'movies')
_trace_dir = os.path.join(_source_dir, 'traces')
_leaf_data = os.path.join(_input_dir, 'B73_WT_vs_rdla_Paired_Rows.csv')
_location_data = os.path.join(_source_dir, 'locations.csv')
_lpy_rays = os.path.join(_source_dir, 'rays.lpy')
_default_axis_up = np.array([0, 0, 1], dtype=np.float64)
_default_axis_x = np.array([1, 0, 0], dtype=np.float64)
_query_options = ['flux_density', 'flux', 'hits', 'areas', 'plantids']
_mesh_format = 'triangle_mesh'
_geom_classes = {
    'ply': PlyDict,
    'obj': ObjDict,
    _mesh_format: PlyDict,
}
_supported_3d_formats = sorted(_geom_classes.keys())
_comm_classes = {
    'ply': PlyFileComm,
    'obj': ObjFileComm,
    _mesh_format: AsciiTableComm,
}
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
_solar_times = ['sunrise', 'noon', 'transit', 'sunset']


class NoDefault(object):
    r"""Stand-in for identifying if a default is passed."""
    pass


class ClassRegistry(object):
    r"""A place to register classes."""

    def __init__(self):
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
        name = getattr(cls, '_name', None)
        if name is not None:
            self._registry.setdefault(key, {})
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
                                 readonly=False):
        r"""Register a cached property.

        Args:
            method (function): Method being registered.
            args (bool, optional): If True, the property will be reset
                when the args for the class are updated.
            readonly (bool, optional): If True, the property can only
                be read, not set.

        """
        classname, methodname = method.__qualname__.rsplit('.', 1)
        registry = self.registry_cached_properties(classname)
        if args:
            registry['args'].append(methodname)
        if readonly:
            dest = registry['readonly']
        else:
            dest = registry['readwrite']
        assert methodname not in dest
        dest.append(methodname)

    def clear_cached_properties(self, instance, preserve=None,
                                cls=None, args=False):
        r"""Clear the cached properties for an instance.

        Args:
            instance (object): Registered class instance to clear the
                cached properties of.
            preserve (list, optional): Set of cached properties that
                should be preserved.
            cls (type, optional): Base class that cached properties
                should be cleared for.
            args (bool, optional): If true, only args cached properties
                should be cleared.

        """
        if preserve is None:
            preserve = []
        registry = self.registry_cached_properties(instance, cls=cls)
        removed = []
        for k in registry['readwrite']:
            if k in preserve or (args and k not in registry['args']):
                continue
            delattr(instance, k)
            removed.append(k)
        for k in registry['readonly']:
            if k in preserve or (args and k not in registry['args']):
                continue
            instance._cached_properties.pop(k, None)
            removed.append(k)
        # if removed:
        #     # pprint.pprint(self._properties)
        #     print(self._instance2str(instance, cls=cls),
        #           "REMOVED", removed,
        #           'bases', instance._registered_base_classes)
        #     pprint.pprint(registry)
        #     pdb.set_trace()
        if cls is None:
            for base in instance._registered_base_classes:
                self.clear_cached_properties(
                    instance, preserve=preserve, cls=base, args=args,
                )

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


def readonly_cached_property(method, args=None):
    r"""Read-only cached property decorator.

    Args:
        method (function): Method who's output should be cached.
        args (bool, optional): If True, the property will be reset when
            the args for the class are updated.

    Returns:
        function: Decorated method.

    """
    _class_registry.register_cached_property(method, args=args,
                                             readonly=True)
    methodname = method.__qualname__.rsplit('.', 1)[-1]

    @property
    def _method_wrapper(self):
        if methodname not in self._cached_properties:
            self._cached_properties[methodname] = method(self)
        return self._cached_properties[methodname]

    return _method_wrapper


def cached_property(method, args=None):
    r"""Cached property decorator.

    Args:
        method (function): Method who's output should be cached.
        args (bool, optional): If True, the property will be reset when
            the args for the class are updated.

    Returns:
        function: Decorated method.

    """
    if functools_cached_property is None:
        return readonly_cached_property(method, args=args)
    _class_registry.register_cached_property(method, args=args)
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
    _registered_base_classes = []

    def __init__(self):
        self._cached_properties = {}

    @staticmethod
    def _on_registration(cls):
        import inspect
        base = inspect.getmro(cls)[1]
        if getattr(base, '_registered_base_classes', None) is not None:
            cls._registered_base_classes = (
                base._registered_base_classes + [base])

    @classmethod
    def log_class(cls, message='', prefix=None, border=False,
                  debug=False, source=None):
        r"""Emit a log message.

        Args:
            message (str, optional): Log message.
            prefix (str, optional): Prefix to use.
            border (bool, optional): If True, add a border before and
                after the message.
            source (str, optional): Source class that the message was
                emitted by.

        """
        if prefix is None:
            if source is None:
                prefix = f'{cls._name}: '
            else:
                prefix = f'{cls._name} [{source}]: '
        msg = f'{prefix}{message}'
        if border:
            line = 80 * '-'
            msg = line + '\n' + msg + '\n' + line
        print(msg)
        if debug:
            pdb.set_trace()

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
        source = None
        if cls is None:
            cls = self
        else:
            source = self._name
        cls.log_class(message=message, source=source, **kwargs)

    def error(self, error_cls, message='', debug=False):
        r"""Raise an error, adding context to the message.

        Args:
            error_cls (type): Error class.
            message (str, optional): Error message.
            debug (bool, optional): If True, set a debug break point.

        """
        prefix = f'{self.log_prefix_instance}: {self.log_prefix_stack}'
        msg = f'{prefix}{message}'
        if debug:
            self.debug(f'{error_cls}({msg})')
        raise error_cls(msg)

    def debug(self, message='', **kwargs):
        r"""Set a pdb break point if debugging is active.

        Args:
            message (str, optional): Log message to show before setting
                break point.
            **kwargs: Additional keyword arguments are passed to the
                log method if it is called.

        """
        self.log(f'DEBUG: {message}', force=True, **kwargs)
        pdb.set_trace()

    def clear_cached_properties(self, preserve=None, args=False):
        r"""Clear the cached properties.

        Args:
            preserve (list, optional): Set of cached properties that
                should be preserved.
            args (bool, optional): If true, only args cached properties
                should be cleared.

        """
        _class_registry.clear_cached_properties(self, preserve=preserve,
                                                args=args)

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
        self.clear_cached_properties(preserve=preserve, args=args)
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
        self.clear_cached_properties(preserve=preserve, args=args)
        self.set_cached_properties(properties)


############################################################
# Methods for I/O
############################################################

def generate_filename(basefile, ext=None, suffix=None, directory=None):
    r"""Generate a filename using the base name from another file.

    Args:
        basefile (str): Base file name that new file should be based on.
        ext (str, optional): File extension that should be used for the
            generated file name.
        suffix (str, optional): Suffix that should be added to the base
            file name in the generated file name.
        directory (str, optional): Directory that the generated file name
            should be in if different than the base file name.

    Returns:
        str: New file name.

    """
    assert ext or suffix or directory
    out = basefile
    if ext:
        out = os.path.splitext(out)[0] + ext
    if suffix:
        out = suffix.join(os.path.splitext(out))
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
        print(f'Writing movie with {len(frames)} frames to \"{fname}\"')
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
        args += [
            '-r', str(frame_rate), '-f', 'concat',
            '-i', os.path.basename(fname_concat), fname,
        ]
        if verbose:
            print(args)
        subprocess.check_call(args, cwd=frame_dir)
    finally:
        if os.path.isfile(fname_concat):
            os.remove(fname_concat)
    if verbose:
        print(f'Wrote movie with {len(frames)} frames to \"{fname}\"')


def read_csv(fname, select=None, verbose=False):
    r"""Read data from a CSV file.

    Args:
        fname (str): Path to file that should be read.
        select (str, list, optional): One or more fields that should be
            selected.
        verbose (bool, optional): If True, log messages will be emitted.

    Returns:
        dict: CSV contents with colums as key/value pairs.

    """
    if verbose:
        print(f'Reading CSV from \"{fname}\"')
    df = pd.read_csv(fname)
    out = {}
    for k in df.columns:
        out[k] = df[k].to_numpy()
    for k in list(out.keys()):
        if ' (' in k:
            k_name, k_units = k.split(' (')
            k_units = k_units.rstrip(')')
            out[k_name] = out.pop(k)
            # out[k_name] = units.QuantityArray(out.pop(k), k_units)
    if isinstance(select, str):
        out = out[select]
    elif isinstance(select, list):
        out = {k: out[k] for k in select}
    if verbose:
        print(f'Read CSV from \"{fname}\"')
    return out


def write_csv(data, fname, verbose=False):
    r"""Write columns to a CSV file.

    Args:
        data (dict): Table columns.
        fname (str): Path to file where the CSV should be saved.
        verbose (bool, optional): If True, log messages will be emitted.

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
    df.to_csv(fname, index=False, header=header)
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
    out = imageio.rmread(fname)
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
    verbose = True
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
            abort = True
            break
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


def read_locations(fname, verbose=False):
    r"""Read location data from a CSV file.

    Args:
        fname (str): Path to file containing location data.
        verbose (bool, optional): If True, log messages will be emitted.

    Returns:
        dict: Mapping between location name and location parameters.

    """
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


def scene2geom(scene, cls, d=None, verbose=False, **kwargs):
    r"""Convert a PlantGL scene to a 3D geometry mesh.

    Args:
        scene (plantgl.Scene): Scene to convert.
        cls (type, str): Name of the type of mesh that should be returned
            or the dictionary class that should be created.
        d (plantgl.Tesselator, optional): PlantGL discretizer.
        verbose (bool, optional): If True, display log messages about
            tasks.
        **kwargs: Additional keyword arguments are passed to shape2dict
            for each shape in the scene.

    Returns:
        cls: 3D geometry mesh for the scene.

    """
    if d is None:
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
    for k, shapes in scene_dict.items():
        for shape in shapes:
            d.clear()
            shapedict = shape2dict(shape, d=d, as_obj=as_obj,
                                   verbose=verbose, **kwargs)
            igeom = cls.from_dict(shapedict)
            if igeom is not None:
                igeom = prune_empty_faces(igeom)
                out.append(igeom)
            d.clear()
    if verbose:
        print('scene2geom: Finished converting scene')
    return out


def shape2dict(shape, d=None, conversion=1.0, as_obj=False,
               color=(0, 255, 0), verbose=False,
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
        if d.result.isColorIndexListToDefault():
            for i, c in enumerate(d.result.colorList):
                for k in ['red', 'green', 'blue']:
                    out['vertices'][i][k] = getattr(c, k)
        else:  # pragma: debug
            raise Exception("Indexed vertex colors not supported.")
    elif color:
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
    if isinstance(mesh, ObjDict):
        mesh_dict['face'] -= 1
    axis_y = np.cross(axis_up, axis_x)
    mesh_dict['vertex'] += x * axis_x
    mesh_dict['vertex'] += y * axis_y
    if plantid and plantids_in_blue:
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

def parse_units(x):
    r"""Parse a units string.

    Args:
        x (str): String containing units.

    Returns:
        units.Units: Units instance.

    """
    return units.Units(x)


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


def parse_solar_time(x, date, latitude, longitude, altitude=None,
                     location=None, method='spa', horizon_buffer=5.0):
    r"""Parse an input string as a solar time.

    Args:
        x (str): Input string. Can be 'sunrise', 'noon', or 'sunset'.
        date (datetime.datetime): Date to get solar time on.
        latitude (float): Latitude of location to get solar time for.
        longitude (float): Longitude of location to get solar time for.
        altitude (float, str): Altitude of location to get solar time
            for.
        location (pvlib.location.Location, optional): pvlib location to
            use instead of creating one.
        method (str, optional): Method that pvlib should use to determine
            the solar times.
        horizon_buffer (float, optional): Time (in minutes) that should
            be added or subtracted to times when the sun is at the
            horizon.

    Returns:
        datetime.datetime: Time determined from solar position.

    """
    import pvlib
    assert x in _solar_times
    if x == 'noon':
        x = 'transit'
    if location is None:
        location = pvlib.location.Location(
            latitude, longitude, altitude=altitude, tz=str(date.tzinfo),
        )
    date_pv = pd.DatetimeIndex([date])
    # date_pv = pvlib.tools._datetimelike_scalar_to_datetimeindex(date)
    out_pd = location.get_sun_rise_set_transit(date_pv, method=method)[x]
    out = out_pd.iloc[0].to_pydatetime()
    if horizon_buffer:
        if x == 'sunrise':
            out += timedelta(minutes=parse_quantity(horizon_buffer,
                                                    'minutes').value)
        elif x == 'sunset':
            out -= timedelta(minutes=parse_quantity(horizon_buffer,
                                                    'minutes').value)
    return out


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


class InstrumentedParser(argparse.ArgumentParser):
    r"""Class for parsing arguments allowing arguments to be
    added to multiple subparsers."""

    def __init__(self, *args, **kwargs):
        self._quantity_units = {}
        self._subparser_mutually_exclusive_groups = {}
        self._subparser_classes = {}
        self._subparser_defaults = {}
        self._subparser_dependencies = {}
        self._subparser_func = kwargs.pop('func', None)
        super(InstrumentedParser, self).__init__(*args, **kwargs)

    @property
    def _primary_subparser(self):
        if len(self._subparser_classes) == 1:
            return list(self._subparser_classes.keys())[0]
        return None

    def parse_args(self, args=None, **kwargs):
        r"""Parse arguments.

        Args:
            args (list, optional): Arguments to parse. Defaults to
                sys.argv if not provided.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        Returns:
            argparse.Namespace: Parsed arguments.

        """
        args0, unparsed = self.parse_known_args()
        args_supp = []
        for k, v in self._subparser_defaults.items():
            if v is None or getattr(args0, k, None):
                continue
            args_supp.append(v)
        if args_supp:
            if args is None:
                args = sys.argv
            args = args_supp + args[1:]
        out = super(InstrumentedParser, self).parse_args(
            args=args, **kwargs)
        out = self.add_units(out)
        return out

    def add_units(self, args):
        r"""Add units to arguments that units were recorded for.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            argparse.Namespace: Parsed arguments.

        """
        for k, v in self._quantity_units.items():
            if hasattr(args, k):
                setattr(args, k, parse_quantity(getattr(args, k), v))
        for k, v in self._subparser_classes.items():
            for vv in v.choices.values():
                vv.add_units(args)
        return args

    def has_subparsers(self, name):
        r"""Check if there is a subparsers group associated with the
        specified name.

        Args:
            name (str): Name of the subparsers group to check for.

        Returns:
            bool: True if the named subparsers group exists.

        """
        return (name in self._subparser_classes)

    def get_subparsers(self, name, default=NoDefault):
        r"""Get the subparsers object with the specified name.

        Args:
            name (str): Name of the subparsers instance to retrieve.
            default (object, optional): Default to return if the
                subparsers instance does not exist.

        Returns:
            InstrumentedSubparsers: Subparsers instance.

        Raises:
            KeyError: If default not provided and the subparsers object
                does not exist.

        """
        out = default
        if name in self._subparser_classes:
            out = InstrumentedSubparsers(
                name, self._subparser_classes[name], self)
        if out is NoDefault:
            raise KeyError(name)
        return out

    def add_subparsers(self, *args, **kwargs):
        r"""Add a subparsers group to this parser.

        Args:
            *args, **kwargs: Arguments are passed to the base class's
                method.

        Returns:
            InstrumentedSubparsers: Wrapped subparsers group.

        """
        name = kwargs['dest']
        self._subparser_dependencies[name] = {}
        self._subparser_defaults[name] = kwargs.pop('default', None)
        kwargs.setdefault('parser_class', InstrumentedParser)
        out = super(InstrumentedParser, self).add_subparsers(
            *args, **kwargs)
        self._subparser_classes[name] = out
        return InstrumentedSubparsers(name, out, self)

    def run_subparser(self, name, args):
        r"""Run the subparser selected by a subparser group.

        Args:
            name (str): Name of the subparser group to run.
            args (argparse.Namespace): Parsed arguments.

        """
        subparser = getattr(args, name)
        if subparser is None:
            subparser = self._subparser_defaults[name]
        func = self._subparser_classes[name].choices[
            subparser]._subparser_func
        if func is None:
            func = _class_registry.get(name, subparser)
        print(f"RUNNING {subparser}")
        func(args)

    def _handle_local_options(self, args, kwargs):
        if 'units' in kwargs:
            kwargs = copy.deepcopy(kwargs)
            self._quantity_units[kwargs['dest']] = kwargs.pop('units')
        return args, kwargs

    def _iter_subparsers(self, args, kwargs, include_name=False):
        subparsers = kwargs.pop('subparsers', None)
        subparser_options = kwargs.pop('subparser_options', {})
        subparser_specific_dest = kwargs.pop(
            'subparser_specific_dest', False)
        kwargs.setdefault('dest', args[0].lstrip('-').replace('-', '_'))
        if subparsers is False:
            assert not subparser_specific_dest
            args, kwargs = self._handle_local_options(args, kwargs)
            if include_name:
                yield None, super(InstrumentedParser, self), args, kwargs
            else:
                yield super(InstrumentedParser, self), args, kwargs
            return
        if (not isinstance(subparsers, dict)) and self._primary_subparser:
            subparsers = {self._primary_subparser: subparsers}
        if not subparsers:
            args, kwargs = self._handle_local_options(args, kwargs)
            if include_name:
                yield None, super(InstrumentedParser, self), args, kwargs
            else:
                yield super(InstrumentedParser, self), args, kwargs
            return
        if not subparsers:
            print(self._primary_subparser)
            pdb.set_trace()
        for k, v in subparsers.items():
            if v is None:
                unique_prog = []
                v = []
                if k in self._subparser_classes:
                    for kk, vv in self._subparser_classes[k].choices.items():
                        if vv.prog not in unique_prog:
                            unique_prog.append(vv.prog)
                            v.append(kk)
                if not v:
                    raise ValueError(f"Could not locate subparser or "
                                     f"group for \"{k}\"")
            for x in v:
                iargs = copy.deepcopy(args)
                ikwargs = dict(kwargs, **subparser_options.get(x, {}))
                if subparser_specific_dest:
                    ikwargs['dest'] += f'_{x}'
                    iargs = tuple(list(iargs) + [args[0] + f'-{x}'])
                if x in self._subparser_mutually_exclusive_groups:
                    xinst = self._subparser_mutually_exclusive_groups[x]
                else:
                    xinst = self._subparser_classes[k].choices[x]
                if include_name:
                    yield (x, xinst, iargs, ikwargs)
                else:
                    yield (xinst, iargs, ikwargs)

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        for x, iargs, ikwargs in self._iter_subparsers(args, kwargs):
            x.add_argument(*iargs, **ikwargs)

    def add_mutually_exclusive_group(self, name, *args, **kwargs):
        r"""Add a mutually exclusive group of arguments.

        Args:
            name (str): Name used to identify the mutually exclusive
                group.
            *args, **kwargs: Additional arguments are passed to the
                add_mutually_exclusive_group method of the subparsers
                (if there are any) or the parent class (if there are not
                any subparsers).

        Returns:
            InstrumentedParserGroup: Wrapped exclusive argument group.

        """
        group = {}
        for k, x, iargs, ikwargs in self._iter_subparsers(
                args, kwargs, include_name=True):
            if isinstance(x, InstrumentedParser):
                group[k] = x.add_mutually_exclusive_group(name)
            else:
                group[k] = x.add_mutually_exclusive_group()
        out = InstrumentedParserGroup(group, parent=self)
        self._subparser_mutually_exclusive_groups[name] = out
        return out


class InstrumentedSubparsers(object):
    r"""Light wrapper for a group of subparsers.

    Args:
        name (str): ID string for the group.
        subparsers (ArgumentParser): Group parser for subparsers.
        parent (ArgumentParser): Parser used to create the subparser
            group.

    """

    def __init__(self, name, subparsers, parent):
        self.name = name
        self._subparsers = subparsers
        self._parent = parent

    def add_parser(self, name, *args, **kwargs):
        r"""Add a parser to the subparser group.

        Args:
            name (str): ID string for the parser.
            *args, **kwargs: Additional arguments are passed to the
                add_parser method for the underlying subparser group.

        Returns:
            ArgumentParser: New subparser.

        """
        self._parent._subparser_dependencies[self.name][name] = kwargs.pop(
            'dependencies', [])
        return self._subparsers.add_parser(name, *args, **kwargs)


class InstrumentedParserGroup(object):
    r"""Simple wrapper for a set of parsers.

    Args:
        members (dict): Set of parsers.
        parent (ArgumentParser, optional): Parser containing this group.

    """

    def __init__(self, members, parent=None):
        self.members = members
        self.parent = parent

    def _iter_subparsers(self, args, kwargs, include_name=False):
        subparsers = kwargs.pop('subparsers', list(self.members.keys()))
        subparser_options = kwargs.pop('subparser_options', {})
        assert isinstance(subparsers, list)
        for k, v in self.members.items():
            if k not in subparsers:
                continue
            ikw = dict(kwargs, **subparser_options.get(k, {}))
            if include_name:
                yield (k, v, args, ikw)
            else:
                yield v, args, ikw

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        for x, iargs, ikwargs in self._iter_subparsers(args, kwargs):
            x.add_argument(*iargs, **ikwargs)


class SubparserBase(RegisteredClassBase):
    r"""Base class for tasks associated with subparsers.

    Args:
        args (argparse.Namespace, optional): Parsed arguments. If not
            provided, additional keyword arguments are parsed to create
            args and keyword arguments that are not used by the parser
            are passed to the run method.
        **kwargs: Additional keyword arguments are passed to the run
            method.

    """

    _name = None
    _help = None
    _default = None
    _arguments = []
    _arguments_suffix_ignore = []
    _argument_modifications = {}
    _excluded_arguments = []
    _excluded_arguments_defaults = {}
    _external_arguments = {}

    def __init__(self, args=None, **kwargs):
        super(SubparserBase, self).__init__()
        if args is None:
            kwargs.setdefault(self._registry_key, self._name)
            args, kwargs = parse_args(**kwargs)
        self._cached_args = []
        self.adjust_args(args)
        self.args = args
        kwargs.setdefault('dont_load_existing', True)
        self.run(**kwargs)

    @staticmethod
    def _on_registration(cls):
        RegisteredClassBase._on_registration(cls)
        if cls._registry_key is None:
            return
        import inspect
        base = inspect.getmro(cls)[1]
        base_args = base.argument_dict(use_flags=True)
        local_args = cls.argument_dict(use_flags=True)
        arguments = copy.deepcopy(base_args)
        subparser_specific = [
            k for k, v in base_args.items()
            if v[1].get('subparser_specific_dest', False)
        ]
        cls._excluded_arguments_defaults = dict(
            cls._excluded_arguments_defaults,
            **{
                base.arg2dest(*base_args[k]): base.arg2default(*base_args[k])
                for k in cls._excluded_arguments + subparser_specific
            }
        )
        for k in cls._excluded_arguments:
            arguments.pop(k)
        arguments.update(**local_args)
        for k, v in cls._argument_modifications.items():
            arguments[k][1].update(**v)
        for kext, mods in cls._external_arguments.items():
            ext_arguments = _class_registry.get(
                cls._registry_key, kext).argument_dict(use_flags=True)
            for k, v in mods.items():
                arguments[k] = copy.deepcopy(ext_arguments[k])
                arguments[k][1].update(**v)
        cls._arguments = list(arguments.values())
        cls._argument_modifications = {}
        cls._excluded_arguments = []
        cls._external_arguments = {}

    @classmethod
    def add_arguments(cls, parser):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.

        """
        if not parser.has_subparsers(cls._registry_key):
            kws = {'title': cls._registry_key,
                   'dest': cls._registry_key}
            if cls._default is not None:
                kws['default'] = cls._default
            parser.add_subparsers(**kws)
        subparsers = parser.get_subparsers(cls._registry_key)
        subparsers.add_parser(cls._name, help=cls._help)
        for iargs, ikwargs in cls._arguments:
            ikwargs = dict(ikwargs, subparsers=[cls._name])
            parser.add_argument(*iargs, **ikwargs)

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        pass

    @classmethod
    def arg2dest(cls, iargs, ikwargs):
        r"""Determine the name that will be used to store an argument.

        Args:
            iargs (tuple): Argument args for add_argument.
            ikwargs (dict): Argument kwargs for add_argument.

        Returns:
            str: Name that argument will be stored under.

        """
        ikey = ikwargs.get(
            'dest', iargs[0].lstrip('--').replace('-', '_'))
        if ikwargs.get('subparser_specific_dest', False):
            ikey += f'_{cls._name}'
        return ikey

    @classmethod
    def arg2default(cls, iargs, ikwargs):
        r"""Determine the default value for an argument.

        Args:
            iargs (tuple): Argument args for add_argument.
            ikwargs (dict): Argument kwargs for add_argument.

        Returns:
            object: Default value for the argument.

        """
        if ikwargs.get('action', None) == 'store_true':
            ival = False
        elif ikwargs.get('action', None) == 'store_false':
            ival = True
        else:
            ival = ikwargs.get('default', None)
        return ival

    @classmethod
    def argument_dict(cls, use_flags=False):
        r"""Get a dictionary of argument data.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            dict: Positional and keyword arguments for creating the
                arguments.

        """
        out = OrderedDict()
        for iargs, ikwargs in cls._arguments:
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = cls.arg2dest(iargs, ikwargs)
            out[ikey] = [iargs, ikwargs]
        return out

    @classmethod
    def argument_defaults(cls, use_flags=False):
        r"""Get a dictionary of argument defaults.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            dict: Argument defaults.

        """
        parser = InstrumentedParser('for defaults')
        cls.add_arguments(parser)
        args = parser.parse_args([cls._name])
        args._in_argument_defaults = True
        cls.adjust_args(args)
        out = {}
        for iargs, ikwargs in cls._arguments:
            dest = cls.arg2dest(iargs, ikwargs)
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = dest
            out[ikey] = getattr(args, dest)
        return out

    @classmethod
    def argument_names(cls, use_flags=False):
        r"""Set of arguments allowed by the subparser.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            list: Arguments.

        """
        out = []
        for iargs, ikwargs in cls._arguments:
            if use_flags:
                out.append(iargs[0])
            else:
                out.append(cls.arg2dest(iargs, ikwargs))
        return out

    @classmethod
    def select_valid_arguments(cls, args, use_flags=False):
        r"""Select arguments from a list that are valid for this class.

        Args:
            args (list, dict): Set of argument names.
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        """
        names = cls.argument_names(use_flags=use_flags)
        return [k for k in args if k in names]

    def copy_args(self, args):
        r"""Copy arguments from the namespace to this instance as
        attributes.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        for k in self.argument_names():
            assert not hasattr(self, k)
            setattr(self, k, getattr(args, k))

    def cache_args(self, adjust=None, args_preserve=None,
                   args_overwrite=None, properties_preserve=None,
                   alternate_outputs=None, recursive=None):
        r"""Cache the current set of arguments.

        Args:
            adjust (type, optional): Class that should be used to adjust
                the arguments after they are updated. If not provided,
                this class will be used.
            args_preserve (list, optional): Set of argument names to
                preserve from the updated argument namespace when the
                cached args are restored (by restore_args).
            args_ovewrite (dict, optional): Argument values to set for
                the run after copying the current argument namespace.
            properties_preserve (list, optional): Set of cached
                properties that should be preserved.
            alternate_outputs (list, optional): Set of alternate outputs
                that should be generated.
            recursive (bool, optional): If True, this is a recursive
                call and overwrite for outputs should be reset to False.
                If not provided, recursive will be set to true if self
                is an instance of the provided adjust class.

        """
        if adjust is None:
            adjust = self
        if args_preserve is None:
            args_preserve = []
        if args_overwrite is None:
            args_overwrite = {}
        if properties_preserve is None:
            properties_preserve = []
        if alternate_outputs is None:
            alternate_outputs = []
        if recursive is None:
            recursive = (adjust._name == self._name)
        enabled_outputs = adjust.enabled_outputs(self.args)
        for name in adjust.output_names():
            output_key = f'output_{name}'
            if name in enabled_outputs + alternate_outputs:
                args_overwrite.setdefault(output_key, True)
            else:
                args_overwrite.setdefault(output_key, False)
            if recursive:
                args_overwrite.setdefault(f'overwrite_{name}', False)
        # TODO: cache/restore cached properties?
        cached_properties = self.pop_cached_properties(
            preserve=properties_preserve, args=True)
        self._cached_args.append(
            (self.args, args_preserve,
             cached_properties, properties_preserve)
        )
        self.args = copy.deepcopy(self.args)
        self.log(f'Updating args with {args_overwrite}')
        for k, v in args_overwrite.items():
            setattr(self.args, k, v)
        adjust.adjust_args(self.args)

    def restore_args(self):
        r"""Restore the last set of preserved arguments."""
        new_args = self.args
        cache_entry = self._cached_args.pop()
        self.args = cache_entry[0]
        for k in cache_entry[1]:
            setattr(self.args, k, getattr(new_args, k))
        cached_properties = cache_entry[2]
        props_preserve = cache_entry[3]
        if props_preserve:
            cached_properties.update(
                self.get_cached_properties(include=props_preserve)
            )
        self.reset_cached_properties(cached_properties, args=True)

    def run(self):
        r"""Run the process associated with this subparser."""
        raise NotImplementedError


class TaskBase(SubparserBase):
    r"""Base class for tasks."""

    _registry_key = 'task'
    _default = 'generate'
    _ext = None
    _output_dir = None
    _alternate_outputs_write_optional = []
    _alternate_outputs_write_required = []
    _arguments_suffix_ignore = [
        'overwrite_all', 'verbose', 'debug',
    ]
    _arguments = [
        (('--output', ), {
            'type': str,
            'help': 'File where output should be saved',
            'subparser_specific_dest': True,
        }),
        (('--overwrite', ), {
            'action': 'store_true',
            'help': 'Overwrite existing output',
            'subparser_specific_dest': True,
        }),
        (('--overwrite-all', ), {
            'action': 'store_true',
            'help': 'Overwrite all child components of the task',
        }),
        (('--verbose', ), {
            'action': 'store_true',
            'help': 'Show log messages'
        }),
        (('--debug', ), {
            'action': 'store_true',
            'help': ('Run in debug mode, setting break points for debug '
                     'messages and errors')
        }),
    ]

    def __init__(self, *args, **kwargs):
        self._alternate_output = {}
        super(TaskBase, self).__init__(*args, **kwargs)

    @staticmethod
    def _on_registration(cls):
        SubparserBase._on_registration(cls)
        if cls._registry_key is None:
            return
        import inspect
        base = inspect.getmro(cls)[1]
        cls._arguments_suffix_ignore = copy.deepcopy(
            base._arguments_suffix_ignore) + cls._arguments_suffix_ignore
        cls._alternate_outputs_write = (
            cls._alternate_outputs_write_required
            + cls._alternate_outputs_write_optional)

    def output_exists(self, name=None):
        r"""bool: True if the output file exists."""
        if isinstance(name, list):
            return all(self.output_exists(name=k) for k in name)
        if name is None:
            name = self._name
        overwrite = getattr(self.args, f'overwrite_{name}')
        output = getattr(self.args, f'output_{name}')
        return (output and (not overwrite) and os.path.isfile(output))

    @classmethod
    def run_class(cls, self, dont_load_existing=False,
                  args_preserve=None, args_overwrite=None,
                  properties_preserve=None,
                  dont_reset_alternate_output=False,
                  return_alternate_output=False,
                  require_alternate_output=None, **kwargs):
        r"""Run the process associated with this subparser.

        Args:
            self (object): Task instance that is running.
            dont_load_existing (bool, optional): If True, existing output
                will not be loaded.
            args_preserve (list, optional): Set of argument names to
                preserve from the updated arguments following a run.
            args_ovewrite (dict, optional): Argument values to set for
                the run after copying the current argument namespace.
            properties_preserve (list, optional): Set of cached
                properties that should be preserved.
            dont_reset_alternate_output (bool, optional): If True, don't
                reset the dictionary of alternate outputs on return.
            return_alternate_output (str, optional): Name of an
                alternate output that should be output instead of the
                standard output.
            require_alternate_output (list, optional): A list of
                alternate output values that should be recorded. If not,
                provided the default required alternate outputs will be
                used.
            **kwargs: Additional keyword arguments are passed to the
                _run method if it is called.

        Returns:
            object: Generated object.

        """
        output_name = cls._name
        if ((return_alternate_output
             and return_alternate_output in cls._alternate_outputs_write)):
            if require_alternate_output is None:
                require_alternate_output = []
            if return_alternate_output not in require_alternate_output:
                require_alternate_output.append(return_alternate_output)
            output_name = return_alternate_output
            return_alternate_output = False
        self.cache_args(adjust=cls, args_preserve=args_preserve,
                        args_overwrite=args_overwrite,
                        alternate_outputs=require_alternate_output)
        output_names = cls.enabled_outputs(self.args)
        out = None
        if ((self.output_exists(name=output_names)
             and return_alternate_output is False)):
            if dont_load_existing:
                self.log(f'Output already exists and overwrite '
                         f'not set: \"{output_names}\"', cls=cls,
                         force=True)
            else:
                self.log(f'Loading existing output \"{output_names}\"',
                         cls=cls, force=True)
                out = cls.read_output(
                    self,
                    require_alternate_output=require_alternate_output
                )
        else:
            # outputs = {k: getattr(self.args, f'output_{k}') for k in
            #            output_names}
            # self.log(f'outputs = {pprint.pformat(outputs)}')
            out = cls._run(self, **kwargs)
            cls.write_output(self, out)
        self.restore_args()
        if output_name != cls._name:
            self.log(f'Returning alternate value for '
                     f'{output_name}', cls=cls)
            out = self.get_alternate_output(output_name)
        if not dont_reset_alternate_output:
            self._alternate_output.clear()
        return out

    def run(self, **kwargs):
        r"""Run the process associated with this subparser.

        Args:
            **kwargs: Additional keyword arguments are passed to
                run_class.

        Returns:
            object: Generated object.

        """
        return self.run_class(self, **kwargs)

    def add_alternate_output(self, key, value):
        r"""Add an alternate value for return by run_class.

        Args:
            key (str): Name of alternate output.
            value (object): Value of alternate output.

        """
        assert key not in self._alternate_output
        self._alternate_output[key] = value

    def pop_alternate_output(self, key, default=NoDefault,
                             preserve=False):
        r"""Pop an alternate output value from the registry.

        Args:
            key (str): Name of alternate output.
            default (object, optional): Value to return if it is not
                present.
            preserve (bool, optional): If True, don't remove the entry.

        Returns:
            object: Alternate output.

        Raises:
            KeyError: If key is not present and default is not provided.

        """
        if key in self._alternate_output:
            if preserve:
                out = self._alternate_output[key]
            else:
                out = self._alternate_output.pop(key)
        else:
            out = default
        if out is NoDefault:
            raise KeyError(key)
        return out

    def get_alternate_output(self, key, default=NoDefault):
        r"""Get an alternate value for return by run_class.

        Args:
            key (str): Name of alternate output.
            default (object, optional): Value to return if it is not
                present.

        Returns:
            object: Alternate output.

        Raises:
            KeyError: If key is not present and default is not provided.

        """
        return self.pop_alternate_output(key, default=default,
                                         preserve=True)

    @classmethod
    def read_output(cls, self, name=None,
                    require_alternate_output=None):
        r"""Load an output file produced by this task.

        Args:
            self (object): Task instance that is running.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.
            require_alternate_output (list, optional): Set of alternate
                output values that should be read. If not provided,
                the set of required alternate values for the class will
                be used.

        Returns:
            object: Contents of the output file.

        """
        is_base = (name is None)
        if name is None:
            name = cls._name
        outputfile = getattr(self.args, f'output_{name}')
        assert outputfile
        self.log(f'Loading existing output from \"{outputfile}\"',
                 cls=cls)
        out = cls._read_output(self.args, name=name)
        if require_alternate_output is None:
            require_alternate_output = cls._alternate_outputs_write_required
        if is_base and require_alternate_output:
            for k in require_alternate_output:
                koutput = cls.read_output(self, name=k)
                self.add_alternate_output(k, koutput)
        return out

    @classmethod
    def write_output(cls, self, output, name=None):
        r"""Write to an output file.

        Args:
            self (object): Task instance that is running.
            output (object): Output object to write to file.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        is_base = (name is None)
        if name is None:
            name = cls._name
        outputfile = getattr(self.args, f'output_{name}')
        if not outputfile:
            return
        self.log(f'Writing output to \"{outputfile}\"', cls=cls)
        cls._write_output(output, self.args, name=name)
        if is_base:
            for k in cls.enabled_outputs(self.args, no_base=True):
                koutput = self.get_alternate_output(k)
                cls.write_output(self, koutput, name=k)

    @classmethod
    def output_names(cls, only_required=False):
        r"""Get the set of outputs associated with this subparser.

        Args:
            only_required (bool, optional): If True, only include
                required outputs.

        Returns:
            list: Output names.

        """
        out = [cls._name]
        if only_required:
            out += cls._alternate_outputs_write_required
        else:
            out += cls._alternate_outputs_write
        return out

    @classmethod
    def enabled_outputs(cls, args, no_base=False):
        r"""Get the set of outputs enabled by a set of arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            no_base (bool, optional): If True, don't include the default
                output name.

        Returns:
            list: Names of enabled outputs.

        """
        out = []
        if not no_base:
            out += [cls._name]
        out += cls._alternate_outputs_write_required
        out += [
            k for k in cls._alternate_outputs_write_optional
            if getattr(args, f'output_{k}', False)
        ]
        return out

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
        raise NotImplementedError

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        raise NotImplementedError

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
        cls.adjust_args_output(args)
        base_overwrite = getattr(args, f'overwrite_{cls._name}')
        for k in cls._alternate_outputs_write:
            if base_overwrite:
                setattr(args, f'overwrite_{k}', True)
            cls.adjust_args_output(args, name=k)

    @classmethod
    def default_filename_applies(cls, args):
        r"""Check if the default file name can be applied by comparing
        the provided arguments against the assumed defaults.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the default name applies.

        """
        for k, expected in cls.argument_defaults(use_dest=True).items():
            if k in cls._arguments_suffix_ignore:
                continue
            actual = getattr(args, k)
            match = (actual == expected)
            if isinstance(match, np.ndarray):
                match = np.all(match)
            if not match:
                print(f'Mismatch {k}: {actual} vs. {expected}')
                pdb.set_trace()
                return False
        return True

    @classmethod
    def adjust_args_output(cls, args, name=None):
        r"""Adjust the parsed arguments controling output, generating a
        file name from the other arguments if it is not present.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        output_var = f'output_{name}'
        x = getattr(args, output_var)
        if x is False:
            return
        elif not isinstance(x, str):
            if ((getattr(args, '_in_argument_defaults', False)
                 and (not cls.default_filename_applies(args)))):
                setattr(args, output_var, False)
                return
            directory = cls.output_dir(args, name=name)
            base = cls.output_base(args, name=name)
            suffix = cls.output_suffix(args, name=name)
            if suffix is False:
                setattr(args, output_var, False)
                return
            ext = cls.output_ext(args, name=name)
            x = generate_filename(
                base, ext=ext, suffix=suffix, directory=directory,
            )
            setattr(args, output_var, x)
            # cls.log_class(f'Using generated value for {output_var}: '
            #               f'\"{x}\"')
        overwrite_var = f'overwrite_{name}'
        if args.overwrite_all:
            setattr(args, overwrite_var, True)
        # TODO: Remove all dependencies?
        if getattr(args, overwrite_var) and os.path.isfile(x):
            cls.log_class(f'Removing existing {output_var}: \"{x}\"')
            os.remove(x)

    @classmethod
    def output_dir(cls, args, name=None):
        r"""Determine the directory that should be used to generate
        an output file name.

        Args:

            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Directory.

        """
        return cls._output_dir

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        return cls._ext

    @classmethod
    def _run(cls, self):
        raise NotImplementedError


class GenerateTask(TaskBase):
    r"""Class for generating 3D canopies."""

    _name = 'generate'
    _help = 'Generate a canopy mesh'
    _output_dir = _mesh_dir
    _arguments_suffix_ignore = [
        'crop', 'crop_class', 'canopy', 'color',
        'overwrite_lpy_param', 'plantid', 'debug_param',
        'unful_leaves', 'mesh_format', 'overwrite_generate',
        'plot_width', 'output_plantids', 'overwrite_plantids',
    ]
    _alternate_outputs_write_required = ['plantids']
    _convert_to_mesh_units = [
        'plot_length', 'plot_width', 'row_spacing', 'plant_spacing',
        'x', 'y',
    ]
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
            'type': str, 'default': _leaf_data,
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
            'type': str, 'choices': ['WT', 'rdla', 'all'],
            'default': 'WT',
            'help': 'Class to generate geometry for',
        }),
        (('--lpy-input', ), {
            'type': str,
            'default': os.path.join(_source_dir, 'maize.lpy'),
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
        (('--canopy', ), {
            'choices': ['single', 'tile', 'unique'],
            'default': 'single',
            'help': 'Type of canopy to generate a mesh for',
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
        (('--axis-up', ), {
            'type': parse_axis, 'default': 'y',
            'help': 'Axis along which plants should grow within the mesh',
        }),
        (('--axis-rows', ), {
            'type': parse_axis, 'default': 'z',
            'help': 'Axis along which rows should be spaced',
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
            'type': str, 'choices': _supported_3d_formats,
            'help': 'Format that mesh should be saved in',
        }),
        # TODO: Use mesh units in input
        (('--mesh-units', ), {
            'type': parse_units, 'default': units.Units('cm'),
            'help': 'Units that mesh should be output in',
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
        if name == 'plantids':
            return read_csv(outputfile, select='plantids')
        return read_3D(outputfile, file_format=args.mesh_format,
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
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        if name == 'plantids':
            return write_csv({'plantids': output}, outputfile)
        write_3D(output, outputfile, file_format=args.mesh_format,
                 verbose=args.verbose)

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
        if not args.output_plantids:
            args.output_plantids = True
        args.save_all_plantids = False
        if args.canopy == 'single':
            args.nrows = 1
            args.ncols = 1
        else:
            if args.plot_width is None:
                args.plot_width = args.nrows * args.row_spacing
            args.nrows = int(args.plot_width / args.row_spacing)
            args.ncols = int(args.plot_length / args.plant_spacing)
        if not args.output_generate:
            if not args.mesh_format:
                args.mesh_format = 'obj'
        super(GenerateTask, cls).adjust_args(args)
        args.plantids_in_blue = False
        if not args.mesh_format:
            args.mesh_format = get_3D_format(args.output_generate)
        for k in cls._convert_to_color_tuple:
            v = getattr(args, k, None)
            if isinstance(v, str):
                setattr(args, f'{k}_str', v)
                setattr(args, k, parse_color(v, convert_names=True))
        if args.color_str == 'plantids':
            args.plantids_in_blue = True
        if args.lpy_param is None:
            args.lpy_param = os.path.join(
                _param_dir, f'param_{args.crop_class}.json')
        if isinstance(args.lpy_param, str):
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
        if name != 'plantids':
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
            ext = _inv_geom_ext[args.mesh_format]
        return ext

    @classmethod
    def _run(cls, self):
        r"""Run the process associated with this subparser."""
        if self.args.crop_class == 'all':
            mesh = None
            df = MaizeGenerator.load_leaf_data(self.args.leaf_data)
            crop_classes = sorted(list(set(df['Class'])))
            self.log(f'Crop class order: {crop_classes}')
            plantids = self.pop_alternate_output('plantids', None)
            if plantids is None:
                plantids = []
            else:
                plantids = [plantids]
            plantid = self.args.plantid
            x = self.args.x
            y = self.args.y
            for i, crop_class in enumerate(crop_classes):
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
        return shift_mesh(mesh, xo, yo, plantid=plantid,
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
        mesh = scene2geom(
            scene, self.args.mesh_format,
            axis_up=self.args.axis_up,
            axis_x=self.args.axis_rows,
            color=color,
        )
        mesh = scale_mesh(mesh, 1.0,
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


class RayTraceTask(GenerateTask):
    r"""Class for running a ray tracer on a 3D canopy."""

    _name = 'raytrace'
    _time_vars = ['time']
    _hour_defaults = {'time': 12}
    _ext = '.csv'
    _output_dir = _trace_dir
    _arguments_suffix_ignore = [
        'mesh', 'query', 'plantids_in_blue', 'separate_plants',
        'locaton', 'time', 'doy', 'hour', 'year', 'timezone',
        'show_rays', 'output_generate', 'overwrite_generate',
        'overwrite_raytrace', 'highlight', 'output_traced_mesh',
        'overwrite_traced_mesh',
    ]
    _alternate_outputs_write_optional = ['traced_mesh']
    _alternate_outputs_write_required = []
    _convert_to_mesh_units = [
        'ground_height',
        'ray_width', 'ray_length', 'arrow_width',
    ]
    _convert_to_color_tuple = [
        'ray_color',
    ]
    _arguments = [
        (('--mesh', ), {
            'type': str,
            'help': ('Path to a file containing the mesh to raytrace. '
                     'If not provided, one will be generated.'),
        }),
        (('--plantids', ), {
            'type': str,
            'help': ('Path to a file containing plant IDs for the faces '
                     'in the provided mesh.'),
        }),
        (('--raytracer', ), {
            'type': str, 'default': 'hothouse',
            'choices': list(_class_registry.keys('raytracer')),
            'help': 'Name of the ray tracer that should be used.',
        }),
        (('--separate-plants', ), {
            'action': 'store_true',
            'help': ('Track plants as separate components. This requires '
                     'that \"--plantids-in-blue\" is also set.'),
        }),
        (('--nrays', ), {
            'type': int, 'default': 512,
            'help': ('Number of rays that should be cast along each '
                     'dimension'),
        }),
        (('--any-direction', ), {
            'action': 'store_true',
            'help': ('Allow light to be deposited by the ray tracer '
                     'from any direction relative to the surface. If '
                     'not set, only ray intercepting a surface from '
                     'the \"top\" will be counted.'),
        }),
        (('--multibounce', ), {
            'action': 'store_true',
            'help': ('Include multiple bounces when performing the '
                     'trace.'),
        }),
        (('--location', ), {
            'type': str, 'default': 'Champaign',
            'choices': sorted(list(
                read_locations(_location_data).keys())),
            'help': ('Name of a registered location that should be used '
                     'to set the location dependent properties: '
                     'timezone, altitude, longitude, latitude'),
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
        (('--longitude', '--long', ), {
            'type': parse_quantity,
            'default': -88.2434, 'units': 'degrees',
            'help': ('Longitude (in degrees) at which the sun should be '
                     'modeled. Defaults to the longitude of Champaign '
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
        (('--time', '-t', ), {
            'type': str, 'default': '2024-06-17',
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
        (('--output-traced-mesh', ), {
            'nargs': '?', 'const': True, 'default': False,
            'help': ('File where the mesh should be saved with faces '
                     'colored by a ray tracer result. If the flag is '
                     'passed without a file name, a file name will be '
                     'generated.'),
        }),
        (('--overwrite-traced-mesh', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing traced mesh file if '
                     '"--output-traced-mesh" is passed'),
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
                     'mesh, \"areas\" uses the area of each face, and '
                     '\"plantids\" uses the IDs of the plant each face '
                     'belongs to'),
        }),
        # (('--query-units', ), {
        #     'type': str,
        #     'help': ('Units that query should be expressed in. Defaults '
        #              'to Watts for query=\"flux\" and unitless '
        #              'otherwise.'),
        # }),
        (('--show-rays', ), {
            'action': "store_true",
            'help': ('Show the rays in the generated mesh if '
                     '"--output-traced-mesh" is passed.'),
        }),
        (('--ray-color', ), {
            'type': parse_color, 'default': '1.0,0.0,0.0',
            'help': ('Color that should be used for rays when '
                     '"--show-rays" is passed. This should be 3 '
                     'comma separated RGB values expressed as floats in '
                     'the range [0, 1]'),
        }),
        (('--highlight', ), {
            'type': str, 'choices': ['min', 'max'],
            'help': ('Highlight the face with the \"min\" or \"max\" '
                     'query value in the resulting (only valid if '
                     '"--output-traced-mesh" is passed).'),
        }),
        (('--highlight-color', ), {
            'type': parse_color, 'default': '1.0,0.0,1.0',
            'help': ('Color to use for highlighted faces if '
                     '"--highlight" is passed.'),
        }),
        (('--ray-width', ), {
            'type': parse_quantity, 'default': 1.0, 'units': 'cm',
            'help': 'Width of rays drawn when "--show-rays" is passed.',
        }),
        (('--ray-length', ), {
            'type': parse_quantity, 'default': 10.0, 'units': 'cm',
            'help': ('Length of rays drawn when "--show-rays" is '
                     'passed. A negative value will cause the distance '
                     'to the scene to be used for the ray length.'),
        }),
        (('--arrow-width', ), {
            'type': parse_quantity, 'default': 2.0, 'units': 'cm',
            'help': ('Width of arrows of rays drawn when "--show-rays" '
                     'is passed.'),
        }),
        (('--colormap', ), {
            'type': str, 'default': 'YlGn_r',
            'help': ('Name of the matplotlib color map that should be '
                     'used to map query values for each face to colors '
                     'if "--output-traced-mesh" is passed'),
            'subparser_specific_dest': True,
        }),
        (('--color-vmin', ), {
            'type': parse_quantity,
            'help': ('Query value that should be mapped to the lowest '
                     'value for the colormap if "--output-traced-mesh" '
                     'is passed'),
            'subparser_specific_dest': True,
        }),
        (('--color-vmax', ), {
            'type': parse_quantity,
            'help': ('Query value that should be mapped to the highest '
                     'value for the colormap if "--output-traced-mesh" '
                     'is passed'),
            'subparser_specific_dest': True,
        }),
        (('--colormap-scaling', ), {
            'type': str, 'choices': ['linear', 'log'],
            'default': 'linear',
            'help': ('Scaling that should be used to map query values '
                     'to colors in the color map if '
                     '"--output-traced-mesh" is passed'),
            'subparser_specific_dest': True,
        }),
        (('--output-generate', ), {
            'type': str,
            'help': ('Path to file where generated mesh should be saved '
                     'other than the default if \"mesh\" is not '
                     'provided.'),
        }),
        (('--overwrite-generate', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing generated mesh if \"mesh\" '
                     'is not provided.'),
        }),
        (('--plantids-in-blue', ), {
            'action': 'store_true',
            'help': 'Plant IDs are stored in the blue color channel. ',
        }),
    ]
    _argument_modifications = {
        '--output': {
            'help': ('File where the flux values for each face in the '
                     'mesh should be saved'),
        },
        '--mesh-format': {
            'help': ('Format that provided \"mesh\" is in or the format '
                     'that the generate mesh should be in if \"mesh\" '
                     'is not provided. If \"--mesh-format\" is not '
                     'provided, the file extension will be used to '
                     'determine the format'),
        },
        '--mesh-units': {
            'help': ('Units that the provided \"mesh\" is in or '
                     'the units the generated mesh should be in if '
                     '\"mesh\" is not provided'),
        },
        '--canopy': {
            'default': 'unique',
        },
    }

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
        if name == 'traced_mesh':
            return read_3D(outputfile, file_format=args.mesh_format,
                           verbose=args.verbose)
        return read_csv(outputfile, verbose=args.verbose)

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
        if name == 'traced_mesh':
            return write_3D(output, outputfile,
                            file_format=args.mesh_format,
                            verbose=args.verbose)
        write_csv(output, outputfile, verbose=args.verbose)

    @property
    def verbose(self):
        r"""bool: Turn on log messages."""
        return self.args.verbose

    @readonly_cached_args_property
    def raytracer(self):
        r"""RayTracerBase: Ray tracer."""
        print("Re-creating ray tracer", self.args.time)
        return _class_registry.get(
            'raytracer', self.args.raytracer)(
                self.args, self.mesh, plantids=self.plantids)

    @cached_property
    def mesh(self):
        r"""ObjDict: Mesh that will be ray traced."""
        return read_3D(self.args.mesh,
                       file_format=self.args.mesh_format,
                       verbose=self.args.verbose)

    @cached_property
    def plantids(self):
        r"""np.ndarray: Plant IDs for each face in the mesh."""
        if os.path.isfile(self.args.plantids):
            return read_csv(
                self.args.plantids,
                verbose=self.args.verbose,
                select='plantids',
            )
        return None

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        if args.mesh is None:
            GenerateTask.adjust_args(args)
            args.mesh = args.output_generate
            args.plantids = args.output_plantids
        if args.location:
            location_data = read_locations(_location_data)
            for k, v in location_data[args.location].items():
                setattr(args, k, v)
        if not (args.pressure or args.altitude):
            args.altitude = parse_quantity(10.0, 'meters')
        cls.adjust_args_time(args)
        super(RayTraceTask, cls).adjust_args(args)

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
        if x in _solar_times:
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
        if x_solar in _solar_times:
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
                x_str = x.isoformat().replace(':', '-')
            setattr(args, timevar, x)
            setattr(args, f'{timevar}_str', x_str)
            # print(f'Updated {timevar} to {x} ({x_str})')

    @classmethod
    def output_dir(cls, args, name=None):
        r"""Determine the directory that should be used to generate
        an output file name.

        Args:

            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Directory.

        """
        if name == 'traced_mesh':
            return _traced_mesh_dir
        return super(RayTraceTask, cls).output_dir(args, name=name)

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
        if name == 'traced_mesh':
            return args.output_raytrace
        return args.mesh

    @classmethod
    def output_suffix(cls, args, name=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            str: Suffix.

        """
        if name == 'traced_mesh':
            suffix = f'_{args.query}'
            if getattr(args, 'show_rays', False):
                suffix += '_rays'
            if isinstance(getattr(args, 'highlight', False), str):
                suffix += f'_highlight{args.highlight.title()}'
            return suffix
        suffix = ''
        if args.location:
            suffix += f"_{args.location}"
        else:
            return False
        suffix += cls.output_suffix_time(args)
        suffix += f'_{args.nrays}'
        if args.multibounce:
            suffix += '_multibounce'
        if args.any_direction:
            suffix += '_anydirection'
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
            return '_'.join(suffixes)
        time_str = getattr(args, f'{timevar}_str', None)
        if time_str:
            return time_str
        time = getattr(args, timevar).replace(microsecond=0)
        return time.isoformat().replace(':', '-')

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
        if name == 'traced_mesh':
            return os.path.splitext(args.mesh)[-1]
        return super(RayTraceTask, cls).output_ext(args, name=name)

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
            face_values = query_values[query]
        else:
            face_values = (
                query_values['flux_density'] * query_values['areas']
            )
        return face_values

    @classmethod
    def query_limits(cls, query_values):
        r"""Compute limits from the query values.

        Args:
            query_values (dict): Query values for each face.

        Returns:
            dict: Limits on each query.

        """
        out = {}
        for query in _query_options:
            out[query] = {}
            values = cls.extract_query(query_values, query)
            if isinstance(values, units.QuantityArray):
                values = values.value
            out[query].update(
                vmin_linear=values[values >= 0].min(),
                vmin_log=values[values > 0].min(),
                vmax_linear=values.max(),
                vmax_log=values.max(),
            )
        return out

    # TODO: This should only be modified if the date changes
    @cached_property
    def color_limits_noon(self):
        r"""tuple: Min/max for the queried values at noon."""
        assert not self.args.time_str.endswith('noon')
        print('GETTING MIN/MAX FROM NOON', self.args.time_str)
        query_values = RayTraceTask.run_class(
            self, args_overwrite={'time': 'noon'},
            return_alternate_output='limits',
        )
        return self.query_limits(query_values)

    @classmethod
    def _set_color_limits(cls, self, query_values, name=None):
        r"""Set the minimum and maximum for color mapping.

        Args:
            self (object): Task instance that is running.
            query_values (dict): Query values for each face.
            name (str, optional): Name for limits to set if different
                than raytrace.

        """
        if name is None:
            name = cls._name
        var_min = f'color_vmin_{name}'
        var_max = f'color_vmax_{name}'
        var_scaling = f'colormap_scaling_{name}'
        if ((getattr(self.args, var_min) is not None
             and getattr(self.args, var_max) is not None)):
            return
        if self.args.time_str.endswith('noon'):
            self.color_limits_noon = self.query_limits(query_values)
        vscaling = getattr(self.args, var_scaling)
        limits = self.color_limits_noon[self.args.query]
        if getattr(self.args, var_min) is None:
            setattr(self.args, var_min, limits[f'vmin_{vscaling}'])
        if getattr(self.args, var_max) is None:
            setattr(self.args, var_max, limits[f'vmax_{vscaling}'])
        # self.log(f'LIMITS[{cls._name}, {name}]: '
        #          f'{getattr(self.args, var_min)}, '
        #          f'{getattr(self.args, var_max)}')

    @classmethod
    def _color_scene(cls, self, query_values):
        r"""Run the ray tracer on the selected geometry.

        Args:
            self (object): Task instance that is running.
            query_values (dict): Query values for each face.

        Returns:
            ObjDict: Generated mesh with ray traced colors.

        """
        face_values = cls.extract_query(query_values, self.args.query)
        if self.args.show_rays:
            mesh = cls.run_class(
                self,
                args_overwrite={'show_rays': False},
                properties_preserve=['raytracer'],
                return_alternate_output='traced_mesh',
            )
            mesh.append(
                generate_rays(self.raytracer.ray_origins,
                              self.raytracer.ray_directions,
                              ray_length=self.raytracer.ray_lengths,
                              geom_format=type(mesh),
                              ray_color=self.args.ray_color,
                              ray_width=self.args.ray_width.value,
                              arrow_width=self.args.arrow_width.value)
            )
            return mesh
        cls._set_color_limits(self, query_values)
        mesh = self.mesh
        vertex_values = self.raytracer.face2vertex(
            face_values, method='deposit')
        vertex_colors = apply_color_map(
            vertex_values,
            color_map=self.args.colormap_raytrace,
            vmin=self.args.color_vmin_raytrace,
            vmax=self.args.color_vmax_raytrace,
            scaling=self.args.colormap_scaling_raytrace,
            highlight=self.args.highlight,
            highlight_color=self.args.highlight_color,
        )
        mesh.add_colors('vertex', vertex_colors)
        return mesh

    @classmethod
    def _raytrace_scene(cls, self):
        r"""Run the ray tracer on the selected geometry.

        Args:
            self (object): Task instance that is running.

        Returns:
            dict: Dictionary of ray tracer queries.

        """
        values = {}
        for k in _query_options:
            if k == 'flux':  # calculated from flux_density & areas
                continue
            self.cache_args(args_overwrite={'query': k},
                            properties_preserve=['raytracer'],
                            recursive=False)
            values[k] = self.raytracer.raytrace()
            self.restore_args()
        if self.args.output_traced_mesh:
            mesh = cls._color_scene(self, values)
            self.add_alternate_output('traced_mesh', mesh)
        return values

    @classmethod
    def raytrace_totals(cls, self, times=None, **kwargs):
        r"""Run the ray tracer on the selected geometry and compute the
        totals for each plant in the scene.

        Args:
            self (object): Task instance that is running.
            times (list, optional): Set of times to get values for. If
                not provided, only the current time will be used.
            **kwargs: Additional keyword arguments are passed to
                run_class.

        Returns:
            dict: Dictionary of ray tracer query totals.

        """
        if times is not None:
            kwargs.setdefault('args_overwrite', {})
            out = None
            for time in times:
                kwargs['args_overwrite']['time'] = time
                iout = cls.raytrace_totals(self, **kwargs)
                if out is None:
                    out = {k: {i: [] for i in v.keys()}
                           for k, v in iout.items()}
                for k, ids in iout.items():
                    for i, v in ids.items():
                        out[k][i].append(v)
            return out
        values = RayTraceTask.run_class(self, **kwargs)
        values['flux'] = cls.extract_query(values, 'flux')
        plantids = values['plantids']
        plantids_unique = np.unique(plantids)
        out = {k: {'total': values[k].sum()} for k in _query_options}
        for i in plantids_unique:
            idx = (plantids == i)
            for k in _query_options:
                out[k][i] = values[k][idx].sum()
        return out

    @classmethod
    def _run(cls, self, **kwargs):
        r"""Run the process associated with this subparser."""
        if self.args.mesh is None:
            self.mesh = GenerateTask.run_class(
                self,
                args_preserve=['output_generate', 'output_plantids'],
            )
            self.args.mesh = self.args.output_generate
            self.args.plantids = self.args.output_plantids
        return cls._raytrace_scene(self, **kwargs)


class RenderTask(RayTraceTask):
    r"""Class for rendering a 3D canopy."""

    _name = 'render'
    _ext = '.png'
    _output_dir = _image_dir
    _arguments_suffix_ignore = [
        'camera_direction', 'output_raytrace', 'overwrite_raytrace',
        'overwrite_render',
    ]
    _alternate_outputs_write_required = []
    _alternate_outputs_write_optional = []
    _convert_to_mesh_units = [
        'image_width', 'image_height',
    ]
    _convert_to_color_tuple = [
        'background',
    ]
    _arguments = [
        (('--camera-type', ), {
            'type': str,
            'choices': ['projection', 'orthographic'],  # 'spherical'],
            'default': 'projection',
            'help': ('Type of camera that should be used to render the '
                     'scene'),
        }),
        (('--camera-direction', ), {
            'type': str,
            'help': ('Direction that camera should face. If not '
                     'provided, the camera will point to the center of '
                     'the scene from its location.'),
        }),
        (('--camera-fov-width', ), {
            'type': parse_quantity, 'units': 'degrees', 'default': 45.0,
            'help': ('Angular width of the camera\'s field of view (in '
                     'degrees) for a projection camera.'),
        }),
        (('--camera-fov-height', ), {
            'type': parse_quantity, 'units': 'degrees', 'default': 45.0,
            'help': ('Angular height of the camera\'s field of view (in '
                     'degrees) for a projection camera.'),
        }),
        (('--camera-up', ), {
            'type': str,
            'help': ('Up direction for the camera. If not provided, the '
                     'up direction for the scene will be assumed.'),
        }),
        (('--camera-location', ), {
            'type': str,
            'help': ('Location of the camera. If not provided, one will '
                     'be determined that captures the entire scene from '
                     'the provided camera direction. If a direction is '
                     'also not provided, the camera will be centered '
                     'on the center of the scene facing down, '
                     'southeast at a distance that captures the entire '
                     'scene.'),
        }),
        (('--image-nx', ), {
            'type': int,
            'help': ('Number of pixels for the rendered image in the '
                     'horizontal direction. If not provided, but '
                     '--image-ny is provided, the value for --image-nx '
                     'will be determined from --image-ny by assuming '
                     'a constant resolution in both directions. If '
                     'neither are provided, --image-ny defaults to '
                     '1024.'),
        }),
        (('--image-ny', ), {
            'type': int,
            'help': ('Number of pixels for the rendered image in the '
                     'vertical direction. If not provided, but '
                     '--image-nx is provided, the value for --image-ny '
                     'will be determined from --image-nx by assuming '
                     'a constant resolution in both directions. If '
                     'neither are provided, --image-ny defaults to '
                     '1024.'),
        }),
        (('--image-width', ), {
            'type': parse_quantity, 'units': 'cm',
            'help': ('Width of the image (in cm). If not provided, '
                     'the width will be set based on the camera '
                     'position and type such that the entire scene '
                     'is captured.'),
        }),
        (('--image-height', ), {
            'type': parse_quantity, 'units': 'cm',
            'help': ('Height of the image (in cm). If not provided, '
                     'the height will be set based on the camera '
                     'position and type such that the entire scene '
                     'is captured.'),
        }),
        (('--background', ), {
            'type': parse_color, 'default': 'transparent',
            'help': ('Background that should be used for the scene.'),
        }),
        (('--resolution', ), {
            'type': parse_quantity, 'units': 'cm**-1',  # 'default': 5,
            'help': ('Resolution that the scene should be rendered with '
                     'in pixels per centimeter. If provided, any '
                     'values provided for --image-nx and --image-ny '
                     'will be ignored. If not provided, the resolution '
                     'in each direction will be determined by '
                     '--image-nx and --image-ny.'),
        }),
        (('--output-raytrace', ), {
            'action': 'store_false',
            'help': ('Output the raytraced mesh used to render the '
                     'scene'),
        }),
        (('--overwrite-raytrace', ), {
            'action': 'store_true',
            'help': ('Overwrite the output files for the raytraced mesh '
                     'used to render the scene'),
        }),
    ]
    _excluded_arguments = [
        '--show-rays', '--ray-color', '--ray-width', '--ray-length',
        '--arrow-width', '--highlight', '--highlight-color',
    ]
    _argument_modifications = {
        '--output': {
            'help': 'File where the rendered image should be saved',
        },
        '--colormap': {
            'default': 'viridis',
        },
        '--colormap-scaling': {
            'default': 'log',
        },
    }

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
        outputfile = getattr(args, f'output_{cls._name}')
        return read_png(outputfile, verbose=args.verbose)

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        outputfile = getattr(args, f'output_{cls._name}')
        write_png(output, outputfile, verbose=args.verbose)

    @classmethod
    def adjust_args(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.show_rays = False
        if ((args.camera_direction is None
             and args.camera_location is None)):
            args.camera_direction = 'downnortheast'
        super(RenderTask, cls).adjust_args(args, **kwargs)
        # if args.output_raytrace:
        #     # Overwrite is required to force return of face values
        #     args.overwrite_raytrace = True

    # @classmethod
    # def adjust_args_output(cls, args, name=None):
    #     r"""Adjust the parsed arguments controling output, generating a
    #     file name from the other arguments if it is not present.

    #     Args:
    #         args (argparse.Namespace): Parsed arguments.
    #         name (str, optional): Base name for variable to set. Defaults
    #             to the task make.

    #     """
    #     super(RenderTask, cls).adjust_args_output(args, name=name)
    #     if args.output_raytrace:
    #         RayTraceTask.adjust_args_output(args)

    @classmethod
    def output_suffix(cls, args, name=None, **kwargs):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        suffix = super(RenderTask, cls).output_suffix(
            args, name=name, **kwargs)
        suffix += f'_{args.query}'
        if isinstance(args.camera_direction, str):
            suffix += f'_{args.camera_direction}'
        else:
            return False
        if args.camera_type != 'projection':
            suffix += f'_{args.camera_type}'
        return suffix

    @classmethod
    def _render_scene(cls, self):
        r"""Render the scene using a ray tracer.

        Args:
            self (object): Task instance that is running.

        Returns:
            np.ndarray: Pixel color data.

        """
        query_values = RayTraceTask.run_class(
            self, args_overwrite={'query': None},
            properties_preserve=['raytracer'],
        )
        RayTraceTask._set_color_limits(self, query_values, name='render')
        face_values = RayTraceTask.extract_query(
            query_values, self.args.query)
        pixel_values = self.raytracer.render(face_values)
        if isinstance(pixel_values, units.QuantityArray):
            pixel_values = copy.deepcopy(pixel_values.value)
        self.add_alternate_output('pixel_values', pixel_values)
        pixel_values = (pixel_values.T)[::-1, :]
        image = apply_color_map(
            pixel_values,
            color_map=self.args.colormap_render,
            vmin=self.args.color_vmin_render,
            vmax=self.args.color_vmax_render,
            scaling=self.args.colormap_scaling_render,
            highlight=(pixel_values < 0),
            highlight_color=self.args.background,
            include_alpha=(len(self.args.background) == 4)
        )
        return image

    @classmethod
    def _run(cls, self, **kwargs):
        r"""Run the process associated with this subparser."""
        if self.args.mesh is None:
            self.mesh = GenerateTask.run_class(
                self,
                args_preserve=['output_generate', 'output_plantids'],
            )
            self.args.mesh = self.args.output_generate
            self.args.plantids = self.args.output_plantids
        return cls._render_scene(self, **kwargs)


class AnimateTask(RenderTask):
    r"""Class for producing an animation."""

    _name = 'animate'
    _ext = None
    _output_dir = _movie_dir
    _time_vars = ['start_time', 'stop_time']
    _hour_defaults = {}
    _arguments_suffix_ignore = [
        'start_time', 'stop_time', 'movie_format', 'output_render',
        'overwrite_render', 'output_totals', 'overwrite_totals',
    ]
    _alternate_outputs_write_required = []
    _alternate_outputs_write_optional = ['totals']
    _arguments = [
        (('--start-time', ), {
            'type': str, 'default': 'sunrise',
            'help': ('Date time (in any ISO 8601 format) that the '
                     'animation should start at. If not provided, the '
                     'time of sunrise for the selected \"location\" (or '
                     '\"latitude\"/\"longitude\") & \"doy\" will be '
                     'used.'),
        }),
        (('--stop-time', ), {
            'type': str, 'default': 'sunset',
            'help': ('Date time (in any ISO 8601 format) that the '
                     'animation should stop at. If not provided, the '
                     'time of sunset for the selected \"location\" (or '
                     '\"latitude\"/\"longitude\") & \"doy\" will be '
                     'used.'),
        }),
        (('--movie-format', ), {
            'type': str, 'choices': ['mp4', 'mpeg', 'gif'],
            'default': 'gif',
            'help': 'Format that the movie should be output in',
        }),
        (('--frame-count', ), {
            'type': int,
            'help': ('The number of frames that should be generated '
                     'between the animation start and end time. If not '
                     'provided, the number of frames will be determined '
                     'from \"frame_interval\"'),
        }),
        (('--frame-interval', ), {
            'type': parse_quantity, 'units': 'hours',
            'help': ('The time (in hours) that should be used for '
                     'frames in the animation. If not provided, '
                     '\"frame_count\" will be used to calculate the '
                     'frame interval. If \"frame_count\" is not '
                     'provided, a frame interval of 1 hour will be '
                     'used.'),
        }),
        (('--frame-rate', ), {
            'type': int, 'default': 1,
            'help': ('The frame rate that should be used for the '
                     'generated movie in frames per second'),
        }),
        (('--output-frames', '--output-render'), {
            'action': 'store_true', 'dest': 'output_render',
            'help': 'Output the rendered frames to disk.',
        }),
        (('--regenerate-frames', '--overwrite-render',
          '--overwrite-frames'), {
            'action': 'store_true', 'dest': 'overwrite_render',
            'help': ('Regenerate frames that already have existing '
                     'images.'),
        }),
        (('--output-totals', ), {
            'nargs': '?', 'const': True, 'default': False,
            'help': ('Output a plot with the totals as a function '
                     'of time.'),
        }),
        (('--overwrite-totals', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing plot of the totals as a '
                     'function of time.'),
        }),
        (('--plot-total', ), {
            'nargs': '?', 'const': 'total',
            'choices': ['total', 'plants'],
            'help': ('Include a plot of the query total below the '
                     'image. If \"--plot-total=plants\" is specified, '
                     'the totals will be shown on a per-plant basis.'),
        }),
    ]
    _argument_modifications = {
        '--output': {
            'help': 'File where the generated animation should be saved',
        },
    }
    _excluded_arguments = [
        '--time',
    ]

    def __init__(self, *args, **kwargs):
        self._figure_totals = None
        self._time_marker = None
        super(AnimateTask, self).__init__(*args, **kwargs)

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
        # outputfile = getattr(args, f'output_{cls._name}')
        raise NotImplementedError

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        outputfile = getattr(args, f'output_{cls._name}')
        write_movie(output, outputfile, frame_rate=args.frame_rate,
                    verbose=args.verbose)

    @classmethod
    def adjust_args(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.output_render = True
        super(AnimateTask, cls).adjust_args(args, **kwargs)
        duration = args.stop_time - args.start_time
        duration = duration.total_seconds() / 3600
        if not args.frame_count:
            if not args.frame_interval:
                args.frame_interval = 1.0
            args.frame_count = int(duration / args.frame_interval)
        elif not args.frame_interval:
            args.frame_interval = duration / args.frame_count
        for k in ['colormap', 'color_vmin', 'color_vmax',
                  'colormap_scaling']:
            setattr(args, f'{k}_render', getattr(args, f'{k}_animate'))

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
        super(AnimateTask, cls).adjust_args_time(args, timevar=timevar)
        if timevar is not None:
            return
        if args.stop_time == args.start_time:
            args.stop_time = args.stop_time.replace(hour=0, minute=0,
                                                    microsecond=0)
            cls.adjust_args_time(args, timevar='stop_time')
        start_time_str = getattr(args, 'start_time_str', None)
        stop_time_str = getattr(args, 'stop_time_str', None)
        if start_time_str and stop_time_str:
            start_parts = start_time_str.rsplit('-', 1)
            stop_parts = stop_time_str.rsplit('-', 1)
            if start_parts[0] == stop_parts[0]:
                args.stop_time_str = stop_parts[-1]
        if not hasattr(args, 'time'):
            args.time = None
            args.time_str = None

    @classmethod
    def output_suffix(cls, args, name=None, **kwargs):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        suffix = super(AnimateTask, cls).output_suffix(
            args, name=name, **kwargs)
        if args.plot_total:
            suffix += '_totals'
        if name != 'totals' and args.frame_rate != 1:
            suffix += f'_{args.frame_rate}fps'
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
        if name == 'totals':
            return '.png'
        return f'.{args.movie_format}'

    @property
    def times(self):
        r"""list: Set of times for frames in the animation."""
        dt = timedelta(hours=self.args.frame_interval)
        time = self.args.start_time
        out = []
        for i in range(self.args.frame_count):
            out.append(time)
            time += dt
        return out

    @property
    def figure_totals(self):
        r"""Figure containing the query totals."""
        if self._figure_totals is not None:
            return self._figure_totals
        import matplotlib.pyplot as plt
        times = self.times
        fig = plt.figure()
        ax = fig.add_subplot(111)
        totals = RayTraceTask.raytrace_totals(self, times=times)[
            self.args.query]
        ylabel = self.args.query.title()  # TODO: Add units
        if self.args.plot_total == 'plants':
            for k, v in totals.items():
                ax.plot(times, v, label=k)
        else:
            ax.plot(times, totals['total'])
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        self._figure_totals = fig
        self._time_marker = ax.axvline(x=self.start_time,
                                       color=(1, 1, 1), alpha=0.5,
                                       linewidth=10)
        return self._figure_totals

    @property
    def time_marker(self):
        r"""matplotlib.lines.line2D: Vertical line marking the time."""
        if self._time_marker is None:
            self.figure_totals
        return self._time_marker

    def add_totals_to_frame(self, frame, time):
        r"""Add a plot of query totals to a frame.

        Args:
            frame (str): Frame containing the rendered scene that the
                plot should be added to.
            time (datetime.datetime): Time that frame is associated with.

        Returns:
            str: Updated frame with the plot added.

        """
        frame_new = '_totals'.join(os.path.splitext(frame))
        if self._figure_totals is None:
            old_data = read_png(frame, verbose=self.args.verbose)
            print('OLD_IMAGE', old_data.shape)
            pdb.set_trace()
            width_px = old_data.shape[0]
            height_px = int(0.2 * width_px)
            dpi = self.figure_totals.get_dpi()
            self.figure_totals.set_size_inches(width_px / dpi,
                                               height_px / dpi)
        fig = self.figure_totals
        self.time_marker.set_xdata([time, time])
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        print('NEW_IMAGE', data.shape)
        pdb.set_trace()
        data_new = np.concatenate([old_data, data])
        print('CONCAT', data_new)
        write_png(data_new, frame_new, verbose=self.args.verbose)
        return frame_new

    @classmethod
    def _run(cls, self):
        r"""Run the process associated with this subparser."""
        frames = []
        for time in self.times:
            # self.cache_args(
            #     args_overwrite={
            #         'time': time.isoformat(),
            #         'time_str': None,
            #         'output_render': True,
            #     },
            #     adjust=RenderTask
            # )
            # iframe = self.args.output_render
            # print('time', time, iframe)
            RenderTask.run_class(self, dont_load_existing=True,
                                 args_overwrite={
                                     'time': time.isoformat(),
                                     'time_str': None,
                                     'output_render': True,
                                 },
                                 args_preserve=['output_render'])
            iframe = self.args.output_render
            print('time', time, iframe)
            # self.restore_args()
            if self.args.plot_total:
                iframe = self.add_totals_to_frame(iframe, time)
            frames.append(iframe)
        if not os.path.isdir(_movie_dir):
            os.mkdir(_movie_dir)
        return frames


############################################################
# LPy parametrization class
#  - 'age' indicates the time since germination
#  - 'n' indicates the phytomer count
############################################################


class PlantParameterBase(RegisteredClassBase):

    _registry_key = 'plant_parameter'


def DelayedPlantParameter(name):

    def parameter_class():
        return _class_registry.get('plant_parameter', name)

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
            kcls = _class_registry.get('plant_parameter', k)
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


#################################################################
# Ray tracing tools
#################################################################

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
            tz=str(self.time.tzinfo),
        )
        self.time_pv = pvlib.tools._datetimelike_scalar_to_datetimeindex(
            self.time)
        self._solar_position = None
        self._irradiance = None

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
                dni_extra=self.dni_extra,  # .value,
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
    param = {
        'ORIGINS': ray_origins.astype(np.float64),
        'DIRECTIONS': ray_directions.astype(np.float64),
        'RAY_WIDTH': ray_width,
        'RAY_LENGTH': ray_length,
        'ARROW_WIDTH': arrow_width,
    }
    lsys = Lsystem(_lpy_rays, param)
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
    return mesh


class RayTracerBase(RegisteredClassBase):
    r"""Base class for ray tracers."""

    _registry_key = 'raytracer'
    _area_min = np.finfo(np.float32).resolution

    def __init__(self, args, mesh, plantids=None):
        super(RayTracerBase, self).__init__()
        self.args = args
        self.mesh = mesh
        self.plantids_face = plantids
        self.verbose = self.args.verbose
        self.mesh_dict = self.mesh.as_array_dict()
        if isinstance(self.mesh, ObjDict):
            self.mesh_dict['face'] -= 1
        self.areas = np.array(self.mesh.areas)
        self.area_mask = (self.areas > self._area_min)
        # print(f'{np.logical_not(self.area_mask).sum()} '
        #       f'faces have areas of 0')
        # self.areas = parse_quantity(self.areas, self.args.mesh_units**2)
        if self.args.plantids_in_blue:
            self.plantids_vertex = (
                255 * self.mesh_dict['vertex_colors'][:, 2]
            ).astype('uint8')
        else:
            self.plantids_vertex = np.zeros(
                (self.mesh_dict['vertex'].shape[0], ), dtype=np.uint8
            )
        self.plants = {}
        if self.args.separate_plants:
            if self.plantids_face is not None:
                for plantid in np.unique(self.plantids_face):
                    if self.args.verbose:
                        print(f'Selecting plant w/ ID \"{plantid}\"')
                    self.plants[plantid] = self.select_faces(
                        self.mesh_dict,
                        (self.plantids_face == plantid),
                        continuous=True,
                    )
            else:
                for plantid in np.unique(self.plantids_vertex):
                    if self.args.verbose:
                        print(f'Selecting plant w/ ID \"{plantid}\"')
                    self.plants[plantid] = self.select_vertices(
                        self.mesh_dict,
                        (self.plantids_vertex == plantid),
                        continuous=True,
                    )
        else:
            self.plants[0] = self.mesh_dict
        for plantid in self.plants.keys():
            self.plants[plantid] = self.select_faces(
                self.plants[plantid], self.area_mask,
                dont_prune_vertices=True,
            )
        self.log(f'Creating scene with up = {self.up}, '
                 f'north = {self.north}, '
                 f'east = {self.east}, ground = {self.ground}')

    def parse_axis(self, x):
        r"""Parse axis values that specify direction relative to the
        scene.

        Args:
            x (str): Input string.

        Returns:
            np.ndarray: Axis vector.

        """
        directions = OrderedDict([
            ('up', self.up),
            ('down', -self.up),
            ('north', self.north),
            ('south', -self.north),
            ('east', self.east),
            ('west', -self.east),
        ])
        if isinstance(x, str):
            if x in directions:
                return directions[x]
            composite = []
            xpartial = x
            for k, v in directions.items():
                if xpartial.startswith(k):
                    composite.append(v)
                    xpartial = xpartial.split(k, 1)[-1]
            if composite and not xpartial:
                out = np.mean(np.vstack(composite), axis=0)
                assert len(out) == 3
                out /= np.linalg.norm(out)
                return out
        return parse_axis(x)

    @classmethod
    def assign_face_data(cls, idx, dst, src):
        r"""Assign face values to a destination array based on the
        supplied index.

        Args:
            idx (dict): Dictionary of indices created when selecting a
                subset of a mesh.
            dst (np.ndarray): Array that values should be copied into.
            src (np.ndarray): Array of face values for the current
                selection of faces.

        """
        idx_chain = []
        iidx = idx
        while iidx is not None:
            idx_chain.append(iidx['face'])
            iidx = iidx.get('parent', None)
        idx_dst = np.arange(len(dst), dtype=np.int32)
        for iidx in idx_chain[::-1]:
            assert idx_dst.shape == iidx.shape
            idx_dst = idx_dst[iidx]
        assert idx_dst.shape == src.shape
        dst[idx_dst] = src

    @classmethod
    def select_vertices(cls, mesh_dict, cond, continuous=False):
        r"""Select a subset of the vertices in a mesh.

        Args:
            mesh_dict (dict): Dictionary of mesh properties.
            cond (np.ndarray): Condition that should be used to select
                the vertices.
            continuous (bool, optional): If True, the faces and vertices
                for which the condition is True are continuous.

        Returns:
            dict: Dictionary of mesh properties with only the selected
                vertices.

        """
        if ((len(cond) != mesh_dict['vertex'].shape[0]
             and 'idx' in mesh_dict)):
            cond = cond[mesh_dict['idx']['vertex']]
        assert len(cond) == mesh_dict['vertex'].shape[0]
        # This verifies that the condition is the same for all vertices
        # in each face.
        # for i, face in enumerate(mesh_dict['face']):
        #     cond_face = cond[face]
        #     assert all(cond_face == cond_face[0])
        cond_face = cond[mesh_dict['face'][:, 0]]
        return cls.select_faces(mesh_dict, cond_face,
                                continuous=continuous)

    @classmethod
    def select_faces(cls, mesh_dict, cond, continuous=False,
                     dont_prune_vertices=False):
        r"""Select a subset of the faces in a mesh.

        Args:
            mesh_dict (dict): Dictionary of mesh properties.
            cond (np.ndarray): Condition that should be used to select
                the faces.
            continuous (bool, optional): If True, the faces and vertices
                for which the condition is True are continuous.
            dont_prune_vertices (bool, optional): If True, vertices that
                are not selected are not removed.

        Returns:
            dict: Dictionary of mesh properties with only the selected
                faces.

        """
        out_class = None
        if isinstance(mesh_dict, (ObjDict, PlyDict)):
            out_class = type(mesh_dict)
            mesh = mesh_dict
            mesh_dict = mesh.as_array_dict()
            if isinstance(mesh, ObjDict):
                mesh_dict['face'] -= 1
        if len(cond) != mesh_dict['face'].shape[0] and 'idx' in mesh_dict:
            cond = cond[mesh_dict['idx']['face']]
        assert len(cond) == mesh_dict['face'].shape[0]
        if np.logical_not(cond).sum() == 0:
            out = mesh_dict
            if out_class is not None:
                out.pop('idx')
                out = out_class.from_dict(out)
            return out
        out = {
            'idx': {'face': cond},
            'face': mesh_dict['face'][cond, :],
        }
        if 'idx' in mesh_dict:
            out['idx']['parent'] = mesh_dict['idx']
        nvert = mesh_dict['vertex'].shape[0]
        out['idx']['vertex'] = np.zeros((nvert, ), dtype=bool)
        if dont_prune_vertices:
            out['idx']['vertex'][:] = True
        else:
            out['idx']['vertex'][np.unique(out['face'])] = True
            if continuous:
                # This is for double checking that vertices are continuous
                # idx_vert = out['idx']['vertex']
                # idx_vert2 = out['idx']['vertex'][1:]
                # idx_vert1 = out['idx']['vertex'][:-1]
                # assert all((idx_vert2 - idx_vert1) == 1)
                out['face'] -= out['face'].min()
            else:
                idx_remove = np.where(
                    np.logical_not(out['idx']['vertex']))[0]
                for idx in idx_remove[::-1]:
                    out['face'][out['face'] > idx] -= 1
        for k in ['vertex', 'vertex_colors']:
            if k not in mesh_dict:
                continue
            out[k] = mesh_dict[k][out['idx']['vertex'], :]
        if out_class is not None:
            out.pop('idx')
            out = out_class.from_dict(out)
        return out

    @cached_args_property
    def up(self):
        r"""np.ndarray: Vector direction of up in the scene."""
        return self.args.axis_up.astype("f4")

    @cached_args_property
    def north(self):
        r"""np.ndarray: Vector direction of north in the scene."""
        return self.args.axis_north.astype("f4")

    @cached_args_property
    def east(self):
        r"""np.ndarray: Vector direction of east in the scene."""
        return np.cross(self.north, self.up)

    @cached_args_property
    def scene_mins(self):
        r"""np.ndarray: Minimum scene vertices in each dimension."""
        return self.mesh_dict['vertex'].min(axis=0)

    @cached_args_property
    def scene_maxs(self):
        r"""np.ndarray: Maximum scene vertices in each dimension."""
        return self.mesh_dict['vertex'].max(axis=0)

    @cached_args_property
    def scene_limits(self):
        r"""np.ndarray: Corners of a box containing the scene."""
        limits = np.vstack([self.scene_mins, self.scene_maxs])
        xx, yy, zz = np.meshgrid(limits[:, 0], limits[:, 1], limits[:, 2])
        return np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

    @cached_args_property
    def scene_center(self):
        r"""np.ndarray: Coordinates of the scene's center."""
        return (self.scene_maxs + self.scene_mins) / 2

    @cached_args_property
    def scene_dim(self):
        r"""np.ndarray: Scene's dimensions in each direction."""
        return (self.scene_maxs - self.scene_mins)

    @cached_args_property
    def ground(self):
        r"""np.ndarray: """
        return (
            np.dot(self.scene_dim / 2, self.north) * self.north
            + np.dot(self.scene_dim / 2, self.east) * self.east
            + self.args.ground_height.value * self.up
        )

    @cached_args_property
    def camera_up(self):
        r"""np.ndarray: Unit vector in the camera's up direction."""
        if self.args.camera_up:
            return self.parse_axis(self.args.camera_up)
        vadj = -self.camera_direction
        vadjup = np.dot(vadj, self.up)
        if vadjup == 0.0:
            out = self.up
        else:
            angle = np.arccos(vadjup)
            vhyp = self.up / np.cos(angle)
            out = vhyp - vadj
        out /= np.linalg.norm(out)
        return out

    @cached_args_property
    def camera_direction(self):
        r"""np.ndarray: Unit vector for camera's pointing direction."""
        if self.args.camera_direction:
            return self.parse_axis(self.args.camera_direction)
        if self.args.camera_location is None:
            out = self.parse_axis('downsoutheast')
        else:
            out = self.scene_center - self.camera_location
        out /= np.linalg.norm(out)
        return out

    @cached_args_property
    def camera_right(self):
        r"""np.ndarray: Unit vector for camera's right direction."""
        return np.cross(self.camera_direction, self.camera_up)

    @cached_args_property
    def camera_location(self):
        r"""np.ndarray: Coordinates of the camera."""
        if self.args.camera_location:
            return self.parse_axis(self.args.camera_location)
        fov_width = np.max(np.abs(np.dot(
            self.scene_limits - self.scene_center,
            self.camera_right)))
        camera_distance = np.abs(
            fov_width / np.tan(self.args.camera_fov_width / 2.0))
        if camera_distance < self.clipping_distance.value:
            camera_distance = self.clipping_distance.value
        out = (
            self.scene_center
            - (camera_distance * self.camera_direction)
        )
        if isinstance(out, units.QuantityArray) and out.is_dimensionless():
            out = out.value
        return out

    @cached_args_property
    def camera_distance(self):
        r"""float: Distance between the camera and the scene scenter."""
        return units.Quantity(
            np.linalg.norm(self.scene_center - self.camera_location),
            self.args.mesh_units)

    @cached_args_property
    def image_width(self):
        r"""float: Image width."""
        if self.args.image_width is not None:
            return self.args.image_width
        if self.args.camera_type == 'projection':
            return 2.0 * self.image_distance * np.abs(np.tan(
                self.args.camera_fov_width / 2.0))
        elif self.args.camera_type == 'orthographic':
            return self.camera_scene_dims[0]
        else:
            raise NotImplementedError(f'Default image width for '
                                      f'{self.args.camera_type} camera')

    @cached_args_property
    def image_height(self):
        r"""float: Image height."""
        if self.args.image_height is not None:
            return self.args.image_height
        if self.args.camera_type == 'projection':
            return 2.0 * self.image_distance * np.abs(np.tan(
                self.args.camera_fov_height / 2.0))
        elif self.args.camera_type == 'orthographic':
            return self.camera_scene_dims[1]
        else:
            raise NotImplementedError(f'Default image height for '
                                      f'{self.args.camera_type} camera')

    @cached_args_property
    def camera_scene_dims(self):
        r"""np.ndarray: Scene dimensions parallel to the image plant."""
        return np.array([
            2 * np.max(np.abs(np.dot(
                self.scene_limits - self.camera_location,
                self.camera_right))),
            2 * np.max(np.abs(np.dot(
                self.scene_limits - self.camera_location,
                self.camera_up))),
            2 * np.max(np.abs(np.dot(
                self.scene_limits - self.camera_location,
                self.camera_direction))),
        ])

    @cached_args_property
    def clipping_distance(self):
        r"""float: Maximum distance of any scene limits from the scene
        center along the camera line-of-sight."""
        return units.Quantity(
            np.max(np.abs(np.dot(
                self.scene_limits - self.scene_center,
                self.camera_direction))),
            self.args.mesh_units)

    @cached_args_property
    def image_distance(self):
        r"""float: Distance of the image plane from the camera."""
        if self.args.camera_type == 'projection':
            if self.args.image_width is not None:
                return np.abs(
                    (self.image_width / 2.0)
                    / np.tan(self.args.camera_fov_width / 2.0))
            elif self.args.image_height is not None:
                return np.abs(
                    (self.image_height / 2.0)
                    / np.tan(self.args.camera_fov_height / 2.0))
            out = self.camera_distance - self.clipping_distance
            if out.value < 0:
                out.value = 0.0
            return out
        elif self.args.camera_type == 'orthographic':
            return units.Quantity(0.0, self.args.mesh_units)
        else:
            raise NotImplementedError(f'Default image distance for '
                                      f'{self.args.camera_type} camera')

    @cached_args_property
    def image_center(self):
        r"""np.ndarray: Coordinates of the image's center."""
        return (
            self.camera_location +
            (self.image_distance.value * self.camera_direction)
        )

    @cached_args_property
    def resolution(self):
        r"""float: Number of pixels per cm that image should be rendered
        will."""
        if self.args.resolution is not None:
            return self.resolution
        elif self.args.image_nx is not None:
            return self.image_nx / self.image_width
        return self.image_ny / self.image_height

    @cached_args_property
    def image_nx(self):
        r"""int: Number of pixels in the x direction."""
        if ((self.args.resolution is not None
             or self.args.image_nx is None)):
            return int(self.image_width * self.resolution)
        return self.args.image_nx

    @cached_args_property
    def image_ny(self):
        r"""int: Number of pixels in the y direction."""
        if ((self.args.resolution is not None
             or (self.args.image_ny is None
                 and self.args.image_nx is not None))):
            return int(self.image_height * self.resolution)
        elif self.args.image_ny is not None:
            return self.args.image_ny
        return 2048

    @property
    def ray_origins(self):
        r"""np.ndarray: Coordinates of ray origins."""
        raise NotImplementedError

    @property
    def ray_directions(self):
        r"""np.ndarray: Ray directions."""
        raise NotImplementedError

    @property
    def ray_lengths(self):
        r"""np.ndarray: Ray lengths."""
        raise NotImplementedError

    def raytrace(self):
        r"""Run the ray tracer and get values for each face.

        Returns:
            np.ndarray: Ray tracer results for each face.

        """
        raise NotImplementedError

    def render(self, values):
        r"""Image the scene.

        Args:
            values (np.ndarray): Values on each face that should be used
                when imaging the scene.

        Returns:
            np.ndarray: Ray tracer results for each pixel.

        """
        raise NotImplementedError

    def face2vertex(self, face_scalar, method='average'):
        r"""Convert an array of scalars for each face to an array of
        scalars for each vertex.

        Args:
            face_scalar (np.ndarray): Array of scalars for each face.
            method (str, optional): Method to use to map from face values
                to vertex values.
                    'average': Average over the values for each face that
                        vertices are part of.
                    'deposit': Split the values for each face amongst its
                        vertices additively.

        Returns:
            np.ndarray: Array of scalars for each vertex.

        """
        faces = self.mesh_dict['face'][self.area_mask, :]
        face_scalar = face_scalar[self.area_mask]
        # if face_scalar.shape == self.idx_faces.shape:
        #     face_scalar = face_scalar[self.idx_faces]
        if method == 'deposit':
            face_scalar /= faces.shape[1]
        vertex_scalar = np.zeros((self.mesh_dict['vertex'].shape[0], ))
        try:
            face_scalar = np.tile(face_scalar, (faces.shape[1], 1)).T
        except units.UnitsError:
            face_scalar = units.QuantityArray(
                np.tile(face_scalar.value, (faces.shape[1], 1)).T,
                face_scalar.units)
        assert face_scalar.shape == faces.shape
        for idx, scalar in zip(faces.flatten(), face_scalar.flatten()):
            vertex_scalar[idx] += scalar
        if method == 'average':
            unique, vertex_counts = np.unique(faces, return_counts=True)
            vertex_scalar[unique] /= vertex_counts
        return vertex_scalar

    def update_time(self, time):
        r"""Update the time represented by the ray tracer.

        Args:
            time (datetime.datetime): New time for tracing.

        """
        self.args.time = time


class HothouseRayTracer(RayTracerBase):

    _name = 'hothouse'

    @cached_args_property
    def scene(self):
        r"""hothouse.scene.Scene: Scene containing geometry."""
        from hothouse.plant_model import PlantModel
        from hothouse.scene import Scene
        out = Scene(
            ground=self.ground, up=self.up, north=self.north,
        )
        for plantid, mesh_dict in self.plants.items():
            triangles = []
            for face in mesh_dict['face']:
                triangles.append(mesh_dict['vertex'][face, :])
            triangles = np.array(triangles)
            plant = PlantModel(
                vertices=mesh_dict['vertex'].astype('f4'),
                indices=mesh_dict['face'].astype('i4'),
                attributes=mesh_dict['vertex_colors'].astype('f4'),
                triangles=triangles.astype('f4'),
            )
            out.add_component(plant)
        return out

    @cached_args_property
    def camera_blaster(self):
        r"""hothouse.blaster.OrthographicRayBlaster: Blaster for camera."""
        from hothouse.blaster import (
            ProjectionRayBlaster, OrthographicRayBlaster,
            SphericalRayBlaster)
        camera_classes = {
            'projection': ProjectionRayBlaster,
            'orthographic': OrthographicRayBlaster,
            'spherical': SphericalRayBlaster,
        }
        kws = {}
        if self.args.camera_type == 'projection':
            kws['fov_width'] = self.args.camera_fov_width
            kws['fov_height'] = self.args.camera_fov_height
        rbcls = camera_classes[self.args.camera_type]
        assert self.image_width.value > 0
        print(f'NX = {self.image_nx}, NY = {self.image_ny}')
        camera_blaster = rbcls(
            center=self.image_center.astype("f4"),
            forward=self.camera_direction.astype("f4"),
            up=self.camera_up.astype("f4"),
            width=self.image_width,
            height=self.image_height,
            nx=self.image_nx,
            ny=self.image_ny,
            **kws
        )
        return camera_blaster

    @cached_args_property
    def solar_model(self):
        r"""SolarModel: Model for the sun using pvlib."""
        return SolarModel(
            self.args.latitude, self.args.longitude, self.args.time,
            altitude=self.args.altitude, pressure=self.args.pressure,
            temperature=self.args.temperature,
            # TODO: Allow additional parameters to be passed?
        )

    @cached_args_property
    def solar_blaster(self):
        r"""hothouse.blaster.SolarBlaster: Blaster for sun."""
        # TODO: Add units to parser
        self.log(f"Total PPFD"
                 f"\n   direct = {self.solar_model.ppfd_direct}"
                 f"\n   diffuse = {self.solar_model.ppfd_diffuse}",
                 force=True)
        return self.scene.get_sun_blaster(
            self.args.latitude, self.args.longitude, self.args.time,
            direct_ppfd=self.solar_model.ppfd_direct,
            diffuse_ppfd=self.solar_model.ppfd_diffuse,
            solar_altitude=self.solar_model.apparent_elevation,
            solar_azimuth=self.solar_model.azimuth,
            nx=self.args.nrays, ny=self.args.nrays,
            multibounce=self.args.multibounce,
        )

    @property
    def ray_origins(self):
        r"""np.ndarray: Coordinates of ray origins."""
        return self.ray_properties[0]

    @property
    def ray_directions(self):
        r"""np.ndarray: Ray directions."""
        return self.ray_properties[1]

    @property
    def ray_lengths(self):
        r"""np.ndarray: Ray lengths."""
        return self.ray_properties[2]

    @cached_args_property
    def ray_properties(self):
        r"""tuple: Ray properties."""
        rb = self.scene.get_sun_blaster(
            self.args.latitude, self.args.longitude,
            self.args.time,
            solar_altitude=self.solar_model.apparent_elevation,
            solar_azimuth=self.solar_model.azimuth,
            nx=10, ny=10, multibounce=False,
        )
        ray_origins = rb.origins
        ray_directions = rb.directions
        ray_lengths = self.args.ray_length.value
        if ray_lengths < 0:
            ray_length0 = -ray_lengths
            ray_lengths = rb.compute_distance(self.scene)
            idx_max = (ray_lengths >= max(ray_lengths))
            ray_lengths[idx_max] = ray_length0
        return (ray_origins, ray_directions, ray_lengths)

    def raytrace(self):
        r"""Run the ray tracer and get values for each face.

        Returns:
            np.ndarray: Ray tracer results for each face.

        """
        self.log(f'Running ray tracer to get {self.args.query} for '
                 f't = {self.args.time} with sun '
                 f'light direction: {self.solar_blaster.forward}',
                 border=True, force=True)
        component_values = None
        value_units = None
        if self.args.query == 'flux_density':
            component_values = self.scene.compute_flux_density(
                self.solar_blaster,
                any_direction=self.args.any_direction,
            )
            # value_units = self.solar_model.ppfd_direct.units
        elif self.args.query == 'hits':
            component_values = self.scene.compute_hit_count(
                self.solar_blaster)
        elif self.args.query == 'areas':
            return self.areas
        elif self.args.query == 'plantids':
            if self.plantids_face is not None:
                return self.plantids_face
            return self.plantids_vertex[self.mesh_dict['face'][:, 0]]
        else:
            raise ValueError(f"Unsupported ray tracer query "
                             f"\"{self.args.query}\"")
        values = np.zeros((self.mesh_dict['face'].shape[0], ), np.float64)
        for k, v in component_values.items():
            self.assign_face_data(self.plants[k].get('idx', None),
                                  values, v)
        if value_units:
            values = parse_quantity(values, value_units)
        return values

    def render(self, values, value_miss=-1.0):
        r"""Image the scene.

        Args:
            values (np.ndarray): Values on each face that should be used
                when imaging the scene.
            value_miss (float, optional): Value that should be used for
                pixels that do not hit anything.

        Returns:
            np.ndarray: Ray tracer results for each pixel.

        """
        camera_hits = self.camera_blaster.compute_count(self.scene)
        out = np.zeros(self.image_nx * self.image_ny, "f4")
        if isinstance(values, units.QuantityArray):
            out = units.QuantityArray(out, values.units)
            value_miss = parse_quantity(value_miss, values.units)
        for ci in range(len(self.scene.components)):
            idx_ci = np.where(camera_hits["geomID"] == ci)[0]
            hits = camera_hits["primID"][idx_ci]
            try:
                out[idx_ci[hits >= 0]] += values[hits[hits >= 0]]
            except TypeError:
                pdb.set_trace()
                raise
        out[camera_hits["primID"] < 0] = value_miss
        return out.reshape((self.image_nx, self.image_ny))


#################################################################
# Function for accessing tasks from yggdrasil
#################################################################

def generate(**kwargs):
    r"""Generate a 3D mesh representing one or more crops.

    Args:
        **kwargs: Keyword arguments are passed to GenerateTask.

    Returns:
        ObjDict: Generated mesh.

    """
    return GenerateTask(**kwargs)


def raytrace(**kwargs):
    r"""Run a solar raytracer on a 3D mesh to get the light intercepted by
    each triangle in the mesh. The query result is then used to color the
    mesh triangles.

    Args:
        **kwargs: Keyword arguments are passed to RayTraceTask.

    Returns:
        ObjDict: Mesh with colors set according to the intercepted light.

    """
    return RayTraceTask(**kwargs)


def render(**kwargs):
    r"""Render an image of a ray traced mesh.

    Args:
        **kwargs: Keyword arguments are passed to RenderTask.

    Returns:
        np.ndarray: Image data.

    """
    return RenderTask(**kwargs)


def animate(**kwargs):
    r"""Create an animation by rendering a ray traced mesh for a period
    of time.

    Args:
        **kwargs: Keyword arguments are passed to AnimateTask.

    Returns:
        list: Set of frames in the animation.

    """
    return AnimateTask(**kwargs)


#################################################################
# CLI
#################################################################

def parse_args(**kwargs):
    r"""Parse arguments provided via the command line or keyword
    arguments

    Args:
        **kwargs: If any keyword args are passed, they are parsed
            instead of the command line arguments.

    Returns:
        argparse.Namespace, dict: Argument namespace and keyword
            keyword arguments that were not parsed.

    """
    parser = InstrumentedParser("Generate a 3D maize model")
    for v in _class_registry.values('task'):
        v.add_arguments(parser)
    if kwargs:
        arglist = [kwargs.get('task', _default_task)]
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    for k in list(kwargs.keys()):
        if hasattr(args, k):
            setattr(args, k, kwargs.pop(k))
    parser.run_subparser('task', args)
    return args, kwargs


if __name__ == "__main__":
    args, _ = parse_args()
