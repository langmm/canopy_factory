import pprint
import pdb
import copy
import numpy as np
import pandas as pd
import scipy
from canopy_factory.utils import scale_factor
from canopy_factory.crops.base import (
    PlantGenerator, PlantParameterBase, CurvePlantParameter,
)


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
