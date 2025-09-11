import numpy as np
from canopy_factory.crops.base import PlantGenerator


class DicotGenerator(PlantGenerator):
    r"""Class for generating generic dicot plant geometries."""

    _plant_name = 'dicot'
    _components = dict(
        PlantGenerator._components,
        Leaf={
            'defaults': {
                'NFirst': 1,
                'NPeriod': 2,
                'WMax': 2,
            },
        },
        Petiole={},
        Cotyledon={
            'defaults': {
                'NFirst': 0,
                'NLast': 0,
                'WMax': 2,
            },
        },
        Branch={
            'defaults': {
                'NFirst': 2,
                'NPeriod': 2,
                'WMax': 2,
            },
        },
    )
    _properties = dict(
        PlantGenerator._properties,
        LeavesPerNode={
            'type': 'integer',
            'description': ('Number of leaves that should be distributed '
                            'around each node'),
        },
    )
    _aliases = dict(
        PlantGenerator._aliases,
        LeavesPerNode='LeafWMax',
    )
    _defaults = dict(
        PlantGenerator._defaults,
        NMax=10,
        # Petiole parameters
        PetioleLength=0.5,
        PetioleWidth=0.125,
        # Leaf parameters
        LeafWMax=5,
        LeafMethod='sweep',
        LeafLength=5.0,
        LeafLengthRelStdDev=0.01,
        LeafWidth=1.25,
        LeafWidthRelStdDev=0.01,
        LeafWidthXFunc='interp',
        LeafWidthXFuncXVals=(0, 1),
        LeafWidthXFuncYVals=np.array([
            0.1, 0.30, 0.42, 0.73, 0.88, 1.0, 0.91, 0.76, 0.55, 0,
        ]),
        LeafThickness=0.01,  # relative to leaf width
        LeafAngle=70,
        LeafRotationAngle=360,
        LeafRotationAngleWFunc='linear',
        LeafRotationAngleWFuncSlope=1,
        LeafRotationAngleWFuncIntercept=0.0,
        LeafBend=1.0,
        LeafBendRelStdDev=0.1,
        LeafBendXFuncMethod='LeafBendX',
        # Cotyledon parameters
        CotyledonMethod='sweep',
        CotyledonLength=1.75,
        CotyledonLengthRelStdDev=0.01,
        CotyledonWidth=1.25,
        CotyledonWidthRelStdDev=0.01,
        CotyledonWidthXFunc='interp',
        CotyledonWidthXFuncXVals=(0, 1),
        CotyledonWidthXFuncYVals=np.array([
            0.10, 0.55, 0.76, 0.91, 1.0, 0.91, 0.76, 0.55, 0.10, 0,
        ]),
        CotyledonThickness=0.02,  # relative to leaf width
        CotyledonAngle=90,
        CotyledonRotationAngle=360,
        CotyledonRotationAngleWFunc='linear',
        CotyledonRotationAngleWFuncSlope=1,
        CotyledonRotationAngleWFuncIntercept=0.0,
        # Internode parameters
        InternodeLength=5.0,
        InternodeWidth=0.3,
        InternodeWidthNFunc='linear',
        InternodeWidthNFuncSlope=-0.9,
        InternodeWidthNFuncIntercept=1.0,
        InternodeRotationAngle=45,
        InternodeRotationAngleRelStdDev=0.1,
        # Branch parameters
        BranchAngle=70,
        BranchRotationAngle=360,
        BranchRotationAngleWFunc='linear',
        BranchRotationAngleWFuncSlope=1,
        BranchRotationAngleWFuncIntercept=0.0,
    )

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
        return 20.0 * np.pi * out
