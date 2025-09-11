import numpy as np
from canopy_factory.crops.monocot import MonocotGenerator


class MaizeGenerator(MonocotGenerator):
    r"""Class for generating maize plant geometries."""

    _plant_name = 'maize'
    _defaults = dict(
        MonocotGenerator._defaults,
        data_year='2024',
        id='B73_WT',
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
