import numpy as np
from canopy_factory.crops.base import PlantGenerator


class DicotGenerator(PlantGenerator):
    r"""Class for generating generic dicot plant geometries."""

    _plant_name = 'dicot'
    _components = dict(
        PlantGenerator._components,
        Leaf={
            'defaults': {
                'WMax': 2,
            },
        },
        Petiole={},
        Cotyledon={
            'defaults': {
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
        NMax=5,
        # Petiole parameters
        PetioleLength=1.0,
        PetioleWidth=0.3,
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
            0.27, 0.30, 0.42, 0.73, 0.88, 1.0, 0.91, 0.76, 0.55, 0,
        ]),
        LeafThickness=0.01,  # relative to leaf width
        LeafAngle=80,
        LeafRotationAngle=360,
        LeafRotationAngleWFunc='linear',
        LeafRotationAngleWFuncSlope=1,
        LeafRotationAngleWFuncIntercept=0.0,
        # Cotyledon parameters
        CotyledonWMax=2,
        CotyledonMethod='sweep',
        CotyledonLength=5.0,
        CotyledonLengthRelStdDev=0.01,
        CotyledonWidth=2.0,
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
        InternodeRotationAngle=90,
        InternodeRotationAngleRelStdDev=0.1,
    )
