import os
import pprint
import pdb
import json
import argparse
import numpy as np
import pandas as pd
import scipy
import functools
from openalea.lpy import Lsystem
import openalea.plantgl.all as pgl
import openalea.plantgl.math as pglmath
from openalea.plantgl.math import Vector2, Vector3, Vector4
from openalea.plantgl.scenegraph import NurbsCurve2D, NurbsCurve
from openalea.plantgl.all import Tesselator
from yggdrasil import units
from yggdrasil.serialize.PlySerialize import PlyDict
from yggdrasil.serialize.ObjSerialize import ObjDict
from yggdrasil.communication.PlyFileComm import PlyFileComm
from yggdrasil.communication.ObjFileComm import ObjFileComm
from yggdrasil.communication.AsciiTableComm import AsciiTableComm


_source_dir = os.path.abspath(os.path.dirname(__file__))
_param_dir = os.path.join(_source_dir, 'param')
_output_dir = os.path.join(_source_dir, 'meshes')
_input_dir = os.path.join(_source_dir, 'input')
_image_dir = os.path.join(_source_dir, 'images')
_leaf_data = os.path.join(_input_dir, 'B73_WT_vs_rdla_Paired_Rows.csv')
_geom_classes = {
    'ply': PlyDict,
    'obj': ObjDict,
    'mesh': PlyDict,
}
_comm_classes = {
    'ply': PlyFileComm,
    'obj': ObjFileComm,
    'mesh': AsciiTableComm,
}


############################################################
# Methods for I/O
############################################################

def read_3D(fname, **kwargs):
    r"""Read data from a 3D geometry file.

    Args:
        fname (str): Path to 3D geometry file that should be loaded. The
            file type is determined by inspecting the file extension.
        **kwargs: Additional keyword arguments are passed to the file
            communicator for the identified file type.

    Returns:
        ObjDict: 3D geometry mesh.

    """
    if fname.endswith('.obj'):
        cls = ObjFileComm
    elif fname.endswith('.ply'):
        cls = PlyFileComm
    elif fname.endswith(('.txt', '.mesh')):
        cls = AsciiTableComm
    else:
        raise ValueError(f"Unsupported 3D geometry file extention: "
                         f"\"{fname}\"")
    fd = cls(fname, address=fname, **kwargs)
    flag, data = fd.recv()
    if not flag:
        raise RuntimeError(f"Failed to read 3D geometry from \"{fname}\"")
    pprint.pprint(data)
    pdb.set_trace()
    return data


############################################################
# 3D Geometry manipulation
############################################################


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


def scene2geom(scene, cls, d=None, **kwargs):
    r"""Convert a PlantGL scene to a 3D geometry mesh.

    Args:
        scene (plantgl.Scene): Scene to convert.
        cls (type, str): Name of the type of mesh that should be returned
            or the dictionary class that should be created.
        d (plantgl.Tesselator, optional): PlantGL discretizer.
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
    for k, shapes in scene.todict().items():
        for shape in shapes:
            d.clear()
            shapedict = shape2dict(shape, d=d, as_obj=as_obj, **kwargs)
            igeom = cls.from_dict(shapedict)
            if igeom is not None:
                out.append(igeom)
            d.clear()
    return out


def shape2dict(shape, d=None, conversion=1.0, as_obj=False,
               color=(0.2, 1, 0.2), upaxis='y'):
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
        upaxis (str, optional): Direction that should be considered "up"
            for the shape.

    Returns:
        dict: Dictionary of 3D geometry components.

    """
    if d is None:
        d = Tesselator()
    d.process(shape)
    if d.result is None:
        raise RuntimeError("Discretization failed")
    out = {}
    axismap = {}
    if upaxis != 'z':
        axismap[upaxis] = 'z'
        axismap['z'] = upaxis
    # Vertices
    for p in d.result.pointList:
        new_vert = {}
        for k in ['x', 'y', 'z']:
            new_vert[k] = float(conversion * getattr(p, axismap.get(k, k)))
        out.setdefault('vertices', [])
        out['vertices'].append(new_vert)
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


############################################################
# LPy parametrization class
#  - 'age' indicates the time since germination
#  - 'n' indicates the phytomer count
############################################################


class NoDefault(object):
    r"""Stand-in for identifying if a default is passed."""
    pass


class PlantGenerator(object):
    r"""Base class for generating plants.

    Args:
        seed (int, optional): Seed that should be used for the random
            number generator to make geometries reproducible.
        verbose (bool, optional): If True, log messages will be printed
            describing the generation process.
        **kwargs: Additional keyword arguments are parsed for parameters
            based on the class attributes described below.

    Attributes:
        parameters (dict): Properties parsed from the provided keyword
            arguments.

    ClassAttributes:
        components (list): Plant components.
        parameter_names (list): Name of plant geometry properties that
            will be described as a composite of different factors
            capturing dependencies on things like age, phytomer number,
            etc. that are controlled by other parameters.
        basic_parameters (dict): Name & default value pairs of simple
            plant geometry parameters that are not controlled by any
            other parameters.
        factors (list): Names of factors that will be evaluated for
            parameters in parameter_names.
        optional_suffixes (dict): Name & default value for suffixes that
            when combined with properties in parameter_names create
            optional control parameters.
        component_suffixes (list): Suffixes that when added to components
            create parameters controlling the components.
        dist_defaults (dict): Pairs of distributions and dictionaries
            containing name & default value for suffixes that when
            combined with properties in parameter_names create optional
            control parameters that parameterize different distributions.

    """

    components = [
        'Leaf', 'Internode', 'Branch',
    ]
    parameter_names = [
        'LeafLength', 'LeafWidth', 'BranchAngle', 'RotationAngle',
        'InternodeLength', 'InternodeWidth',
    ]
    basic_parameters = {}
    factors = [
        'EXP', 'VAR', 'N', 'AGE', 'X',
    ]
    optional_suffixes = {
        'Dist': 'normal',
        'Exp': NoDefault,
        'ExpBase': NoDefault,
        'AgeMax': NoDefault,
        'AgeMature': NoDefault,
    }
    component_suffixes = [
        'NMax', 'AgeMax', 'AgeMature',
    ]
    dist_defaults = {
        'normal': {
            'Mean': 1.0,
            'StdDev': 0.2,
        },
        'uniform': {
            'Bounds': (0.0, 1.0),
        },
    }

    def __init__(self, seed=0, verbose=False, **kwargs):
        self.seed = seed
        self.verbose = verbose
        self.parameters = {}
        for k in self.parameter_names:
            self.add_parameter(kwargs, k)
            for suffix, v in self.optional_suffixes.items():
                self.add_parameter(kwargs, f'{k}{suffix}', v)
            dist = self.get_parameter(f'{k}Dist', None)
            if dist is not None:
                for suffix, v in self.dist_defaults.get(
                        dist, {}).items():
                    self.add_parameter(kwargs, f'{k}{suffix}', v)
            if f'{k}Exp' in kwargs:
                if k.endswith('Width'):
                    kwargs.setdefault(
                        f'{k}ExpBase', k.replace('Width', 'Length'))
        for k, v in self.basic_parameters.items():
            self.add_parameter(kwargs, k, v)
        for k in self.components:
            for suffix in self.component_suffixes:
                self.add_parameter(kwargs, f'{k}{suffix}')
        self._key_stack = []
        self._cache = {}
        self._generators = {}
        self.log(f'parameters = {pprint.pformat(self.parameters)}')
        if kwargs:
            pprint.pprint(kwargs)
        assert not kwargs
        super(PlantGenerator, self).__init__(**kwargs)

    def _push_key(self, param, age, n, x):
        key = (param, age, n, x)
        if key in self._cache:
            return self._cache[key]
        self._key_stack.append(key)

    def _pop_key(self, out=None):
        if out is not None:
            self._cache[self._current_key] = out
        self.log(str(out))
        self._key_stack.pop()

    @property
    def _current_key(self):
        if self._key_stack:
            return self._key_stack[-1]
        return None

    @property
    def current_component(self):
        r"""str: Name of component currently being generated."""
        key = self._current_key
        if key is None:
            return None
        for i in range(1, len(key[0])):
            if key[0][i].isupper():
                return key[0][:i]
        raise RuntimeError(f'Could not extract component from \"{key[0]}\"')

    @property
    def generator(self):
        r"""np.random.Generator: Current random number generator."""
        seed = self.seed
        if self._current_key:
            seed += self._current_key[2]
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
        return getattr(self.generator, profile)(*args, **kwargs)

    def __getattr__(self, k):
        if k in self.basic_parameters:
            return self.parameters[k]
        return functools.partial(self.get, k)

    def log(self, message='', force=False):
        r"""Emit a log message.

        Args:
            message (str): Log message.
            force (bool, optional): If True, print the log message even
                if self.verbose is False.

        """
        if not (self.verbose or force):
            return
        prefix = ''
        key = self._current_key
        if key is not None:
            k, age, n, x = key[:]
            if x is None:
                x_str = ''
            else:
                x_str = f',x={x}'
            prefix = f'{k}[{age},{n}{x_str}]: '
        msg = f'{prefix}{message}'
        print(msg)

    def get(self, k, age, n, x=None, override=False):
        r"""Get the value of a complex parameter.

        Args:
            k (str): Name of the parameter.
            age (float): Age of the component that the parameter will be
                used to generate.
            n (int): Phytomer count of the component that the parameter
                will be used to generate.
            x (float, optional): Position along the component that the
                parameter will be used to generate.
            override (bool, optional): If True, any explicit method
                with a name matching k will not be used.

        Returns:
            object: Parameter value.

        """
        out = self._push_key(k, age, n, x)
        if out is not None:
            return out
        if self.hasmethod(f'{k}Param'):
            new_param = getattr(self, f'{k}Param')(age, n, x=x)
            self.log(f'updated parameters = \n'
                     f'{pprint.pformat(new_param)}')
            self.parameters.update(new_param)
        out = self.parameters.get(k, 1.0)
        self.log(f'base = {out}')
        kws = {}
        for factor in self.factors:
            if k.endswith('Curve'):
                self._previous_factor = out
            fout = self.get_factor(factor, k, age, n, x=x, **kws)
            if isinstance(fout, (pgl.NurbsCurve, pgl.NurbsCurve2D)):
                if out != 1.0:
                    raise Exception
                out = fout
            else:
                out *= fout
        self._pop_key(out)
        return out

    def hasmethod(self, k):
        r"""Check if there is an explicit method for a parameter.

        Args:
            k (str): Parameter name.

        Returns:
            bool: True if there is an explicit method, False otherwise.

        """
        return hasattr(type(self), k)

    def add_parameter(self, kwargs, k, default=NoDefault, suffix=None):
        r"""Add a parameter to the parameter attribute from keyword
        arguments.

        Args:
            kwargs (dict): Keyword arguments to extract parameters from.
            k (str): Parameter name to extract.
            default (object, optional): If provided, this value will be
                added to the parameter dictionary if the parameter is not
                present in the provided keyword arguments.
            suffix (str, optional): Suffix that should be added to the
                parameter name before checking for it in kwargs.

        """
        if suffix:
            k = f'{k}{suffix}'
        if k in kwargs:
            self.parameters[k] = kwargs.pop(k)
        elif default != NoDefault:
            self.parameters[k] = default

    def get_parameter(self, k, default=NoDefault, suffix=None):
        r"""Get a parameter from the parameter dictionary.

        Args:
            k (str): Parameter name.
            default (object, optional): If provided, this value will be
                returned if the parameter is not present.
            suffix (str, optional): Suffix that should be added to the
                parameter name before checking for it.

        Returns:
            object: Parameter value.

        Raises:
            KeyError: If the parameter is not present and default is not
                provided.

        """
        names = []
        if suffix:
            ck = self.current_component
            names += [f'{k}{suffix}', f'{ck}{suffix}']
        else:
            names += [k]
        for name in names:
            if name in self.parameters:
                return self.parameters[name]
        if default != NoDefault:
            return default
        raise KeyError(names)

    def get_factor(self, name, *args, **kwargs):
        r"""Get a parameter factor.

        Args:
            name (str): Factor name.
            *args, **kwargs: Additional arguments are passed to the
                factor method.

        Returns:
            object: Factor value.

        """
        func = getattr(self, f'_{name.lower()}_factor')
        out = func(*args, **kwargs)
        if out is None:
            self.log(f'FACTOR[{name}] NOT SET')
            return 1.0
        self.log(f'FACTOR[{name}] = {out}')
        return out

    @classmethod
    def interpolate_points(cls, x, xvals, yvals, **kwargs):
        r"""Interpolate an evolving set of control points to a new value.

        Args:
            x (float): New independent value for which the control
                points be interpolated.
            xvals (np.ndarray, tuple): Independent values that the
                elements in yvals correspond to. If a tuple is provided,
                xvals will be generated as a linear progression from
                the first value to the second.
            yvals (np.ndarray): Array of control points for each value
                in xvals.
            **kwargs: Additional keyword arguments are passed to
                scipy.interpolate.interp1d.

        Returns:
            np.ndarray: Control points at value x.

        """
        if isinstance(xvals, tuple):
            xvals = np.linspace(xvals[0], xvals[1], len(yvals))
        f = scipy.interpolate.interp1d(xvals, yvals, **kwargs)
        return f(x)

    @classmethod
    def create_curve(cls, points, knots=None, uniform=False,
                     stride=60, degree=3, close=False):
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
            close (bool, optional): If True, the curve will be closed.

        Returns:
            NurbsCurve: Curve instance.

        """
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

    def _exp_factor(self, k, age, n, x=None):
        if not (f'{k}Exp' in self.parameters
                and f'{k}ExpBase' in self.parameters):
            return None
        key_cache = (self.parameters[f'{k}ExpBase'], age, n, None)
        if key_cache not in self._cache:
            pdb.set_trace()
        base = self.get(self.parameters[f'{k}ExpBase'], age, n, x=None)
        return np.pow(base, self.parameters[f'{k}Exp'])

    def _var_factor(self, k, age, n, x=None):
        profile = self.get_parameter(f'{k}Dist', None)
        if profile is None:
            return None
        if profile == 'normal':
            return self.generator.normal(
                self.parameters[f'{k}Mean'],
                self.parameters[f'{k}StdDev'],
            )
        elif profile == 'uniform':
            return self.generator.uniform(
                *self.parameters[f'{k}Bounds'],
            )
        else:
            raise ValueError(f"Unsupported profile \"{profile}\"")

    def _x_factor(self, k, age, n, x=None):
        if x is None or not self.hasmethod(f'{k}X'):
            return None
        return getattr(self, f'{k}X')(x)

    def _n_factor(self, k, age, n, x=None):
        param = self.get_parameter(k, None, suffix='NMax')
        if not (self.hasmethod(f'{k}N') and (param is not None)):
            return None
        return getattr(self, f'{k}N')(n / param)

    def _age_factor(self, k, age, n, x=None):
        param = self.get_parameter(k, None, suffix='AgeMature')
        if param is None:
            return None
        if age > param:
            return 1.0
        # TODO: More realistic age dependency?
        # A = np.sqrt(1.0 / param)
        # return A * np.sqrt(age)
        slope = 1.0 / float(param)
        intercept = 0.0
        return (slope * age) + intercept


class MaizeGenerator(PlantGenerator):
    r"""Class for generating maize plant geometries."""

    parameter_names = PlantGenerator.parameter_names + [
        'LeafBend', 'LeafTwist', 'LeafCurve',
    ]
    basic_parameters = dict(
        PlantGenerator.basic_parameters,
        leaf_data_file=None,
        leaf_data_time=27,
        crop_class='WT',
        unfurl_leaves=False,
    )
    leaf_data_parameters = [
        'Length', 'Width', 'Area',
    ]
    _cached_leaf_data = {}

    def __init__(self, **kwargs):
        self._leaf_data = None
        self._leaf_data_analysis = None
        super(MaizeGenerator, self).__init__(**kwargs)

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
        if crop_class not in ['WT', 'rdla', None]:
            print(crop_class)
            pdb.set_trace()
        if key not in cls._cached_leaf_data:
            if crop_class is None:
                print(f"Loading leaf data from \"{fname}\"")
                cls._cached_leaf_data[key] = pd.read_csv(fname)
            else:
                df = cls.load_leaf_data(fname)
                print(f"Selecting {crop_class} from \"{fname}\"")
                df = df.loc[df['Class'] == crop_class]
                # print(df)
                cls._cached_leaf_data[key] = df
        return cls._cached_leaf_data[key]

    @property
    def leaf_data(self):
        r"""pandas.DataFrame: Data contained in the leaf_data_file"""
        if self._leaf_data is None:
            if not self.leaf_data_file:
                raise AttributeError("No leaf data provided")
            # TODO: Scale to correct for time?
            self._leaf_data = self.load_leaf_data(
                self.leaf_data_file,
                crop_class=self.crop_class,
            )
        return self._leaf_data

    @property
    def leaf_data_analysis(self):
        r"""dict: Parameters describing the leaf data."""
        if self._leaf_data_analysis is not None:
            return self._leaf_data_analysis
        nmin = 1
        nmax = 1
        df = self.select_leaf_data(n=nmax)
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
            profile = self.get_parameter(f'{k}Profile', 'normal')
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

    def leaf_param(self, k, n, **kwargs):
        r"""Get updated leaf parameters from the provided leaf data for
        the current phytomer count through interpolation.

        Args:
            k (str): Parameter name.
            n (int): Current phytomer count.
            **kwargs: Additional keyword arguments are provided to
                scipy.interpolate.interp1d.

        Returns:
            dict: Dictionary of updated leaf parameters.

        """
        profile = self.leaf_data_analysis['dists'][k]
        nvals = self.leaf_data_analysis['nvals']
        param = self.leaf_data_analysis['dist_param'][k]
        kwargs.setdefault('axis', 0)
        f = scipy.interpolate.interp1d(nvals, param, **kwargs)
        out = {k: 1.0, f'{k}Exp': 0}
        for suffix, v in zip(self.dist_defaults[profile], f(n)):
            out[f'{k}{suffix}'] = v
        return out

    def LeafLengthParam(self, age, n, x=None):
        r"""Explicit method to update LeafLength control parameters
        from the provided leaf data for the current phytomer count.

        Args:
            age (float): Age of the component that the parameter will be
                used to generate.
            n (int): Phytomer count of the component that the parameter
                will be used to generate.
            x (float, optional): Position along the component that the
                parameter will be used to generate.

        Returns:
            object: Parameter value.

        """
        if not self.leaf_data_file:
            return self.get('LeafLength', age, n, x=x, override=True)
        return self.leaf_param('LeafLength', n)

    def LeafWidthParam(self, age, n, x=None):
        r"""Explicit method to update LeafWidth control parameters
        from the provided leaf data for the current phytomer count.

        Args:
            age (float): Age of the component that the parameter will be
                used to generate.
            n (int): Phytomer count of the component that the parameter
                will be used to generate.
            x (float, optional): Position along the component that the
                parameter will be used to generate.

        Returns:
            object: Parameter value.

        """
        if not self.leaf_data_file:
            return self.get('LeafWidth', age, n, x=x, override=True)
        return self.leaf_param('LeafWidth', n)

    def LeafWidthX(self, x):
        r"""Explicit method to compute the dependence of LeafWidth on
        position along the leaf.

        Args:
            x (float, optional): Position along the leaf that will be
                generated.

        Returns:
            object: Parameter value.

        """
        yvals = np.array(
            [0.09, 0.1, 0.14, 0.24, 0.29, 0.33, 0.3, 0.25, 0.18, 0]
        )
        return self.interpolate_points(x, (0, 1), yvals, axis=0)

    def LeafTwistX(self, x):
        r"""Explicit method to compute the dependence of LeafTwist on
        position along the leaf.

        Args:
            x (float, optional): Position along the leaf that will be
                generated.

        Returns:
            object: Parameter value.

        """
        amp = 0.5
        period = 1.0 / 3.0
        return 2.0 * np.pi * amp * np.sin(2.0 * np.pi * x / period)

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

    def LeafCurveX(self, x):
        r"""Explicit method to compute the dependence of LeafCurve on
        position along the leaf.

        Args:
            x (float, optional): Position along the leaf that will be
                generated.

        Returns:
            object: Parameter value.

        """
        sr2 = np.sqrt(2.0)
        xvals = np.array([0.0, 0.3, 1.0]).T
        yvals = np.array([
            [
                [+0.0,  1.0],
                [-sr2,  sr2],
                [-1.0,  0.0],
                [-sr2, -sr2],
                [+0.0, -1.0],
                [+0.0, -1.0],
                [+sr2, -sr2],
                [+1.0,  0.0],
                [+sr2,  sr2],
                [+0.0,  1.0],
            ],
            [
                [-1.0,  0.2],
                [-0.5,  0.1],
                [-0.2,  0.0],
                [-0.1,  0.0],
                [+0.0,  0.0],
                [+0.0,  0.0],
                [+0.1,  0.0],
                [+0.2,  0.0],
                [+0.5,  0.1],
                [+1.0,  0.2],
            ],
            [
                [-1.0,  0.2],
                [-0.5,  0.1],
                [-0.2,  0.0],
                [-0.1,  0.0],
                [+0.0,  0.0],
                [+0.0,  0.0],
                [+0.1,  0.0],
                [+0.2,  0.0],
                [+0.5,  0.1],
                [+1.0,  0.2],
            ],
        ])
        yvals *= self._previous_factor
        if self.unfurl_leaves:
            pointsX = self.interpolate_points(x, xvals, yvals, axis=0)
        else:
            pointsX = yvals[-1, :]
        return self.create_curve(pointsX)

    def InternodeWidthN(self, x):
        r"""Explicit method to compute the dependence of InternodeWidth
        on phytomer count.

        Args:
            x (float, optional): Normalized phytomer count for the leaf
                that will be generated.

        Returns:
            object: Parameter value.

        """
        intercept = 0.9
        slope = -0.5
        return (slope * x) + intercept

    def BranchAngleN(self, x):
        r"""Explicit method to compute the dependence of BranchAngle
        on phytomer count.

        Args:
            x (float, optional): Normalized phytomer count for the leaf
                that will be generated.

        Returns:
            object: Parameter value.

        """
        intercept = 0.5
        slope = -0.4
        return (slope * x) + intercept


def extract_lpy_param(args):
    r"""Extract parameters for the LPy system from the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: LPy parameters.

    """
    args.overwrite_lpy_param = True
    if args.lpy_param and (os.path.isfile(args.lpy_param)
                           and not args.overwrite_lpy_param):
        with open(args.lpy_param, 'r') as fd:
            out = json.load(fd)
        return out
    generator_args = {
        'leaf_data_file': args.leaf_data,
        'leaf_data_time': args.leaf_data_time,
        'crop_class': args.crop_class,
        'unfurl_leaves': args.unfurl_leaves,
        'verbose': args.verbose,
    }
    out = {
        'MAIZE3D_PARAM': generator_args,
        'NLEAFDIVIDE': args.n_leaf_divide,
        'OUTPUT_TIME': args.time,
    }
    if args.lpy_param:
        with open(args.lpy_param, 'w') as fd:
            json.dump(out, fd)
    return out


def scale_mesh(mesh, scale):
    r"""Scale a 3D mesh.

    Args:
        mesh (ObjDict): Mesh to shift.
        scale (float): Scale factor.

    Returns:
        ObjDict: Scaled mesh.

    """
    mesh_dict = mesh.as_array_dict()
    if isinstance(mesh, ObjDict):
        mesh_dict['face'] -= 1
    mesh_dict['vertex'] *= scale
    return type(mesh).from_array_dict(mesh_dict)


def shift_mesh(mesh, x, y, upaxis='y'):
    r"""Shift a 3D mesh.

    Args:
        mesh (ObjDict): Mesh to shift.
        x (float): Amount to shift the plant in the x direction.
        y (float): Amount to shift the plant in the y direction.
        upaxis (str, optional): Direction that should be considered "up"
            for the mesh. The shifts will be performed perpendicular to
            this direction.

    Returns:
        ObjDict: Shifted mesh.

    """
    mesh_dict = mesh.as_array_dict()
    if isinstance(mesh, ObjDict):
        mesh_dict['face'] -= 1
    if upaxis == 'z':
        idx_x = 0
        idx_y = 1
    elif upaxis == 'x':
        idx_x = 1
        idx_y = 2
    elif upaxis == 'y':
        idx_x = 2
        idx_y = 0
    mesh_dict['vertex'][:, idx_x] += x
    mesh_dict['vertex'][:, idx_y] += y
    out = type(mesh).from_array_dict(mesh_dict)
    return out


def generate_single_plant(args, x=None, y=None, plantid=0):
    r"""Generate a 3D mesh for a single plant.

    Args:
        args (argparse.Namespace): Parsed arguments.
        x (float, optional): Amount to shift the plant in the x direction.
        y (float, optional): Amount to shift the plant in the y direction.
        plantid (int, optional): Plant ID to use as the random number
            generator seed.

    Returns:
        ObjDict: Generated mesh.

    """
    args.lpy_param['MAIZE3D_PARAM']['seed'] = plantid
    lsys = Lsystem(args.lpy_input, args.lpy_param)
    tree = lsys.axiom
    for i in range(args.niter):
        tree = lsys.iterate(tree, 1)
    scene = lsys.sceneInterpretation(tree)
    mesh = scene2geom(
        scene, args.output_format, upaxis=args.upaxis,
        color=args.color,
    )
    if x is not None and y is not None:
        mesh = shift_mesh(mesh, x, y, upaxis=args.upaxis)
    return mesh


def generate_plot(args, x=None, y=None, plantid=0, **kwargs):
    r"""Generate a 3D mesh for a single plot.

    Args:
        args (argparse.Namespace): Parsed arguments.
        x (float, optional): Amount to shift the plot in the x direction.
        y (float, optional): Amount to shift the plot in the y direction.
        plantid (int, optional): Plant ID to use as the random number
            generator seed.
        **kwargs: Additional keyword arguments are passed to calls to
            generate_single_plant.

    Returns:
        ObjDict: Generated mesh.

    """
    generator = np.random.default_rng(seed=plantid)
    if args.lpy_param is None:
        os.path.join(_param_dir, f'param_{args.crop_class}.json')
    args.lpy_param = extract_lpy_param(args)

    if args.canopy:
        if x is None:
            x = 0.0
        if y is None:
            y = 0.0

    # Generate mesh for single plant
    mesh = generate_single_plant(
        args, x=x, y=y, plantid=plantid, **kwargs
    )
    if not args.canopy:
        return mesh

    def posdev():
        return generator.normal(
            0.0, args.location_stddev * args.plant_spacing
        )

    # Generate canopy
    plantid += 1
    nrows = int(args.plot_width / args.row_spacing)
    ncols = int(args.plot_length / args.plant_spacing)
    if args.canopy == 'unique':
        for i in range(nrows):
            ix = x + i * args.row_spacing
            for j in range(ncols):
                iy = y + j * args.plant_spacing
                if i == 0 and j == 0:
                    continue
                mesh.append(
                    generate_single_plant(
                        args, x=(ix + posdev()), y=(iy + posdev()),
                        plantid=plantid, **kwargs
                    )
                )
                plantid += 1
    elif args.canopy == 'tile':
        mesh_single = type(mesh)(mesh)
        for i in range(nrows):
            ix = i * args.row_spacing
            for j in range(ncols):
                iy = j * args.plant_spacing
                if i == 0 and j == 0:
                    continue
                mesh.append(
                    shift_mesh(
                        mesh_single, ix + posdev(), iy + posdev(),
                        upaxis=args.upaxis
                    )
                )
    return mesh


def generate_mesh(args, **kwargs):
    r"""Generate a 3D mesh.

    Args:
        args (argparse.Namespace): Parsed arguments.
        **kwargs: Additional keyword arguments are passed to calls to
            generate_plot.

    Returns:
        ObjDict: Generated mesh.

    """
    add_crop_classes = []
    if args.crop_class == 'all':
        df = MaizeGenerator.load_leaf_data(args.leaf_data)
        crop_classes = sorted(list(set(df['Class'])))
        print(f'Crop class order: {crop_classes}')
        args.lpy_param = None
        args.crop_class = crop_classes[0]
        add_crop_classes += crop_classes[1:]

    mesh = generate_plot(args, **kwargs)
    if not add_crop_classes:
        return mesh

    y = 0.0
    x = 0.0
    if args.canopy:
        nrows = int(args.plot_width / args.row_spacing)
        ncols = int(args.plot_length / args.plant_spacing)
    else:
        nrows = 1
        ncols = 1
    nplants = nrows * ncols
    plantid = 0
    for crop_class in add_crop_classes:
        args.lpy_param = None
        args.crop_class = crop_class
        plantid += nplants
        x += args.row_spacing * (nrows + 2)
        mesh.append(
            generate_plot(
                args, x=x, y=y, plantid=plantid, **kwargs
            )
        )
        # TODO: Labels
    return mesh


def generate(args):
    r"""Generate & output a canopy mesh based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    """
    mesh = generate_mesh(args)
    outkws = {}
    outcls = _comm_classes[args.output_format]
    if args.output_format == 'mesh':
        mesh = mesh.mesh
        outkws['format_str'] = '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'
    out = outcls('mesh', address=args.output, **outkws)
    if args.output_scale != 1.0:
        out = scale_mesh(out, args.output_scale)
    flag = out.send(mesh)
    if not flag:
        raise RuntimeError("Failed to output mesh")
    print(f"Saved mesh to \"{args.output}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a 3D maize model")
    parser.add_argument(
        '--time', type=float, default=27,
        help='Time to generate model for (in days)')
    parser.add_argument(
        '--leaf-data', type=str, default=_leaf_data,
        help='File containing raw leaf data')
    parser.add_argument(
        '--leaf-data-time', type=float, default=27,
        help='Time that data containing raw leaf data was collected')
    parser.add_argument(
        '--n-leaf-divide', type=int, default=10,
        help='Number of segments to divide leaves into')
    parser.add_argument(
        '--crop-class', type=str, choices=['WT', 'rdla', 'all'],
        default='WT',
        help='Class to generate geometry for')
    parser.add_argument(
        '--lpy-input', type=str,
        default=os.path.join(_source_dir, 'maize.lpy'),
        help='File containing LPy L-system rules')
    parser.add_argument(
        '--lpy-param', type=str,
        help='File containing parameters for L-system rules')
    parser.add_argument(
        '--overwrite-lpy-param', action='store_true',
        help='Overwrite the existing lpy_param file')
    parser.add_argument(
        '--niter', type=int, default=20,
        help='Number of iterations to generate')
    parser.add_argument(
        '--canopy', choices=['tile', 'unique'],
        help='Generate a 3D mesh for an entire maize canopy')
    parser.add_argument(
        '--plot-length', type=float, default=200,
        help='Length of plot rows forming canopy (in cm)')
    parser.add_argument(
        '--plot-width', type=float,
        help=('Width of plot forming canopy (in cm). If provided '
              '\'nrows\' will be determined based on the provided '
              '\'row_spacing\'. If not provided, \'plot_width\' will '
              'be determined from \'nrows\' and \'row_spacing\'.'))
    parser.add_argument(
        '--nrows', type=int, default=4,
        help='Number of rows to generate in plot')
    parser.add_argument(
        '--row-spacing', type=float, default=50,
        help='Space between adjacent rows in plot (in cm)')
    parser.add_argument(
        '--plant-spacing', type=float, default=10,
        help='Space between adjacent plants in rows (in cm)')
    parser.add_argument(
        '--color', type=str, default='0.2,1.0,0.2',
        help=('Comma separated RGB values expressed as floats in '
              'range [0, 1]'))
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show log messages')
    parser.add_argument(
        '--upaxis', type=str, default='y',
        help='Axis along which plants should grow')
    parser.add_argument(
        '--unfurl-leaves', action='store_true',
        help='Start leaves as cylinders and then unfurl them')
    parser.add_argument(
        '--location-stddev', type=float, default=0.2,
        help=('Standard deviation relative to \'plant_spacing\' that '
              'should be used when selecting planting locations for '
              'multi-plant canopies'))
    parser.add_argument(
        '--output', type=str,
        help='File where the generated mesh should be saved')
    parser.add_argument(
        '--output-format', type=str, choices=['obj', 'ply', 'mesh'],
        help='Format that mesh should be saved in')
    parser.add_argument(
        '--output-units', type=str, default='cm',
        help='Units that mesh should be output in')
    parser.add_argument(
        '--output-scale', type=float, default=1.0,
        help='Scale factor that should be applied to the output mesh')
    args = parser.parse_args()
    output_format = None
    if args.output_units != 'cm':
        args.output_scale *= float(units.Quantity(1.0, 'cm').to(
            args.output_units))
    if args.plot_width is None:
        args.plot_width = args.nrows * args.row_spacing
    if args.color:
        args.color = tuple([float(x) for x in args.color.split(',')])
    if args.output:
        if args.output.endswith('.obj'):
            output_format = 'obj'
        elif args.output.endswith('.ply'):
            output_format = 'ply'
        elif args.output.endswith(('.txt', '.mesh')):
            output_format = 'mesh'
    else:
        suffix = f'_{args.crop_class}'
        if args.canopy:
            suffix += f'_canopy{args.canopy.title()}'
        if args.unfurl_leaves:
            suffix += '_unfurled'
        if not args.output_format:
            args.output_format = 'obj'
        args.output = os.path.join(
            _output_dir, f'maize{suffix}.{args.output_format}')
    if ((args.output_format and output_format
         and output_format != args.output_format)):
        raise RuntimeError(f'Output format \"{args.output_format}\" '
                           f'does not match file extension on output '
                           f'\"{args.output}\"')
    generate(args)
