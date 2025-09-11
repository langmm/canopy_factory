import os
import pdb
import numpy as np
import pandas as pd
from yggdrasil import units
from canopy_factory import utils
from canopy_factory.crops.base import (
    DictWrapper, DistributionPlantParameter
)
from canopy_factory.crops.monocot import MonocotGenerator


class MaizeGenerator(MonocotGenerator):
    r"""Class for generating maize plant geometries."""

    _plant_name = 'maize'
    _default_data = os.path.join(
        utils.cfg['directories']['input'],
        'B73_WT_vs_rdla_Paired_Rows.csv')
    _defaults = dict(
        MonocotGenerator._defaults,
        id='WT',
        NMax=20,
        LeafAngle=90,
        LeafAngleRelStdDev=0.1,
        LeafAngleAgeFuncAgeMature=4.2,
        LeafAngleNFunc='linear',
        LeafAngleNFuncSlope=-0.4,
        LeafAngleNFuncIntercept=0.5,
        LeafAgeSenesce=27,
        LeafAgeMature=6.3,
        LeafBend=1.0,
        LeafBendRelStdDev=0.1,
        LeafBendXFuncMethod='LeafBendX',
        LeafLength=60.0,
        LeafLengthRelStdDev=0.2,
        LeafUnfurledLength=0.3,   # relative to leaf length
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
        LeafThickness=0.1,  # relative to leaf width
        LeafTwist=1.0,
        LeafTwistRelStdDev=0.1,
        LeafTwistXFunc='sin',
        LeafTwistXFuncAmplitude=(2.0 * np.pi * 0.5),
        LeafTwistXFuncPeriod=(1.0 / 3.0),
        LeafWidthXFunc='interp',
        LeafWidthXFuncXVals=(0, 1),
        LeafWidthXFuncYVals=np.array([
            0.27, 0.30, 0.42, 0.73, 0.88, 1.0, 0.91, 0.76, 0.55, 0,
            # 0.09, 0.1, 0.14, 0.24, 0.29, 0.33, 0.3, 0.25, 0.18, 0
        ]),
        InternodeMethod='cylinder',
        InternodeNDivide=10,
        InternodeLength=8.5,
        InternodeLengthRelStdDev=0.2,
        # InternodeLengthAgeFunc='logistic',
        # InternodeLengthAgeFuncXOffset=0.5,
        # InternodeLengthAgeFuncYOffset=1.0,
        # InternodeLengthAgeFuncAmplitude=(1 + np.exp(-1)),
        InternodeWidthAgeFunc='one',
        InternodeWidthFuncExp=0.15,  # 1.5 in Cieslak
        InternodeWidthFuncVarName='InternodeLength',
        InternodeWidthNFunc='linear',
        InternodeWidthNFuncSlope=-0.5,
        InternodeWidthNFuncIntercept=0.9,
        InternodeAngle=0.0,
        InternodeAngleStdDev=0.5,
        InternodeAgeMature=1.0,
        InternodeWidthXFuncSlopeAgeFuncAgeMature=8.0,
        InternodeRotationAngle=180,
        InternodeRotationAngleRelStdDev=0.1,
    )

    @classmethod
    def ids_from_file(cls, fname):
        r"""Determine all of the available ids from the provided file.

        Args:
            fname (str): Data file.

        Returns:
            list: Crop IDs.

        """
        df = pd.read_csv(fname)
        return sorted(list(set(df['Class'])))

    @classmethod
    def parameters_from_file(cls, args, parameters):
        r"""Calculate parameters based on emperical data.

        Args:
            args (ParsedArguments): Parsed arguments.
            parameters (dict): Parameter dictionary to update.

        Returns:
            dict: Set of parameters calculated from args.

        """
        fname = args.data
        name = args.id
        unit_set = utils.UnitSet.from_attr(args, prefix='data_units_')
        print(f"Loading leaf data from \"{fname}\"")
        df0 = pd.read_csv(fname)
        print(f"Selecting {name} from \"{fname}\"")
        df = df0.loc[df0['Class'] == name]
        if df.empty:
            print(f"No data found for name \"{name}\"")
            print(df0)
            pdb.set_trace()
        # TODO: SCALE UNITS
        nmin = 1
        nmax = 1
        while not cls.select_leaf_data(df, n=nmax).empty:
            nmax += 1
        nvals = np.array(range(nmin, nmax))
        out = {'LeafNMax': nmax}
        params = []
        for col in df.filter(regex=r'^[VR]\d+ '):
            p = col.split(' ', 1)[-1]
            if p not in params:
                params.append(p)
        for p in params:
            kleaf = f'Leaf{p.title()}'
            if kleaf in ['LeafArea']:
                continue
            DictWrapper.remove_prefixed_keys(parameters, f'{kleaf}Func')
            kunits = None
            if unit_set is not None:
                kclass = cls.get_class(kleaf)
                kunits = getattr(unit_set, kclass._unit_dimension)
            profile = getattr(args, f'{kleaf}Dist', None)
            if profile is None:
                profile = 'normal'
            if profile != 'normal':
                out[kleaf] = 1.0
                out[f'{kleaf}Dist'] = profile
            df_p = cls.select_leaf_data(df, parameter=p)
            param_values = DistributionPlantParameter.parametrize_dist(
                df_p, profile=profile, axis=0,
            )
            if profile == 'normal':
                param_values['RelStdDev'] = param_values.pop(
                    'StdDev') / param_values['Mean']
            for kdist, v in param_values.items():
                if kunits is not None and kdist != 'RelStdDev':
                    v = units.QuantityArray(v, kunits)
                if profile != 'normal':
                    kdst = f'{kleaf}Dist{kdist}'
                elif kdist == 'Mean':
                    kdst = kleaf
                else:
                    kdst = f'{kleaf}{kdist}'
                out[f'{kdst}'] = 1.0
                out[f'{kdst}NFunc'] = 'interp'
                out[f'{kdst}NFuncXVals'] = nvals / nmax
                out[f'{kdst}NFuncYVals'] = v
        parameters.update(out)
        return out

    @classmethod
    def select_leaf_data(cls, df, crop_class=None, parameter=None,
                         n=None):
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
        if crop_class is not None:
            df = df.loc[df['Class'] == crop_class]
        if parameter is not None:
            df = df.filter(regex=f' {parameter.title()}$')
        if n is not None:
            assert not (n % 1)
            n = int(n) + 1
            df = df.filter(regex=f'^V{n} ')
        return df

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
