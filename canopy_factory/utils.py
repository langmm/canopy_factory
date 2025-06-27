import os
import pdb
import copy
import numpy as np
import pandas as pd
import functools
import warnings
import subprocess
from datetime import timedelta
import openalea.plantgl.all as pgl
from openalea.plantgl.all import Tesselator
from yggdrasil import rapidjson, units
from yggdrasil.serialize.PlySerialize import PlyDict
from yggdrasil.serialize.ObjSerialize import ObjDict
from yggdrasil.communication.PlyFileComm import PlyFileComm
from yggdrasil.communication.ObjFileComm import ObjFileComm
from yggdrasil.communication.AsciiTableComm import AsciiTableComm


functools_cached_property = getattr(functools, "cached_property", None)
_source_dir = os.path.abspath(os.path.dirname(__file__))
_output_dir = os.path.join(os.getcwd(), 'output')
_input_dir = os.path.join(os.getcwd(), 'input')
_data_dir = os.path.join(_source_dir, 'data')
_lpy_dir = os.path.join(_data_dir, 'lpy')
_param_dir = os.path.join(_source_dir, 'param')
_user_param_dir = os.path.join(_output_dir, 'param')
_location_data = os.path.join(_data_dir, 'locations.csv')
_lpy_rays = os.path.join(_lpy_dir, 'rays.lpy')
_default_axis_up = np.array([0, 0, 1], dtype=np.float64)
_default_axis_x = np.array([1, 0, 0], dtype=np.float64)
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

    def error(self, error_cls, message='', debug=False, prefix=None):
        r"""Raise an error, adding context to the message.

        Args:
            error_cls (type): Error class.
            message (str, optional): Error message.
            debug (bool, optional): If True, set a debug break point.
            prefix (str, optional): Prefix to use.

        """
        if prefix is None:
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
        args_palette = copy.deepcopy(args)
        args_palette += [
            '-f', 'concat', '-i', os.path.basename(fname_concat),
            '-vf', 'palettegen=reserve_transparent=true', 'palette.png',
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
    df = pd.read_csv(fname)
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


def read_locations(fname=None, verbose=False):
    r"""Read location data from a CSV file.

    Args:
        fname (str, optional): Path to file containing location data. If
            not provided _location_data will be used.
        verbose (bool, optional): If True, log messages will be emitted.

    Returns:
        dict: Mapping between location name and location parameters.

    """
    if fname is None:
        fname = _location_data
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
# TODO: Move these into cli.py?

def jsonschema2argument(json):
    r"""Contruct a argparser argument from a jsonschema description.

    Args:
        json (dict): JSON schema.

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
    if typename is not None:
        if not isinstance(typename, str):
            raise TypeError(f'JSON type should be string, not {typename} '
                            f'(json = {json})')
        if ((typename == 'array'
             and isinstance(json.get('items', None), dict))):
            out = jsonschema2argument(json['items'])
            if (('minItems' in json and 'maxItems' in json
                 and json['minItems'] == json['maxItems'])):
                out['nargs'] = json['minItems']
            else:
                out['nargs'] = '+'
        elif typename == 'boolean':
            if json['default']:
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
    if 'default' in json:
        out['default'] = json['default']
    if json.get('type', None) == 'ndarray':
        out['nargs'] = '+'
    if 'description' in json:
        out['help'] = json['description']
    return out


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
            out += timedelta(minutes=int(parse_quantity(horizon_buffer,
                                                        'minutes')))
        elif x == 'sunset':
            out -= timedelta(minutes=int(parse_quantity(horizon_buffer,
                                                        'minutes')))
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

    def convert(self, x):
        r"""Convert a quantity to this unit system.

        Args:
            x (units.QuantityArray): Quantity with units.

        Returns:
            units.QuantityArray: x converted to this unit system.

        """
        assert isinstance(x, units.QuantityArray)
        return x.to_system(self.units)
