import numpy as np
from canopy_factory.crops.dicot import DicotGenerator


class TomatoGenerator(DicotGenerator):
    r"""Class for generating tomato plant geometries."""

    _plant_name = 'tomato'
    _defaults = dict(
        {
            k: v for k, v in DicotGenerator._defaults.items()
            if not k.startswith(('Leaf', 'Petiole', 'Pedicel'))
        },
        NMax=3,
        ApexBLast=1,
        BranchBLast=-1,
        BranchNPeriod=1,
        BranchPlastocron=2,
        NodeSubcomponents=[
            'CotyledonComplex', 'CompoundLeaf', 'FruitCluster',
        ],
        # Plastocron=rapidjson.units.Quantity(1.0, 'days'),
        # Apex
        ApexAngleBFunc='expression',
        ApexAngleBFuncExpression='1.0 if B > 0 else 0.25',
        # Internode
        InternodeLength=8.0,
        InternodeWidth=0.3,
        InternodeAgeMature=3,
        InternodeWidthNFunc='linear',
        InternodeWidthNFuncSlope=-0.7,
        InternodeWidthNFuncIntercept=1.0,
        InternodeLengthBFunc='linear',
        InternodeLengthBFuncSlope=-0.5,
        InternodeLengthBFuncIntercept=1.0,
        # Petiole parameters
        PetioleLength=0.5,
        # Pedicel parameters
        FruitClusterPedicelLength=1.0,
        # Compound leaf
        CompoundLeafAgeMature=6,
        CompoundLeafBpMax=1,  # This is the number of branches/leaf
        CompoundLeafNpMax=5,  # This is the number of nodes/leaf
        CompoundLeafWMax=1,
        CompoundLeafNFirst=1,
        CompoundLeafNPeriod=2,
        CompoundLeafAngle=10,
        CompoundLeafRotationAngle=0,
        CompoundLeafInternodeLength=1.0,
        # CompoundLeafInternodeWidth=0.125,
        CompoundLeafInternodeBend=0.5,
        CompoundLeafInternodeBendRelStdDev=0.1,
        CompoundLeafInternodeBendXFuncMethod='LeafBendX',
        # LeafletComplex
        LeafletComplexAngle=70,
        LeafletComplexAngleNpFunc='expression',
        LeafletComplexAngleNpFuncExpression=(
            '0.0 if Np >= 5 else 1.0'
        ),
        LeafletComplexRotationAngle=90,
        LeafletComplexRotationAngleNpFunc='expression',
        LeafletComplexRotationAngleNpFuncExpression=(
            '1.0 if (Np % 2) == 0 else 3.0'
        ),
        LeafletComplexTiltAngle=90,
        LeafletComplexTiltAngleNpFunc='expression',
        LeafletComplexTiltAngleNpFuncExpression=(
            '3.0 if (Np % 2) == 0 else 1.0'
        ),
        # Leaflet
        LeafletWMax=1,
        LeafletNpFirst=0,
        LeafletNpPeriod=1,
        LeafletMethod='sweep',
        LeafletLength=3.0,
        LeafletLengthRelStdDev=0.01,
        LeafletWidth=1.25,
        LeafletPetioleLength=0.5,
        # LeafletPetioleWidth=0.125,
        LeafletWidthRelStdDev=0.01,
        LeafletWidthXFunc='interp',
        LeafletWidthXFuncXVals=(0, 1),
        LeafletWidthXFuncYVals=np.array([
            0.05, 0.30, 0.42, 0.73, 0.88, 1.0, 0.91, 0.76, 0.55, 0,
        ]),
        LeafletProfileClosed=False,
        LeafletProfileSymmetry=[0],
        LeafletProfileReverse=True,
        LeafletProfileControlPoints=np.array([
            [+0.0,  0.0],
            [+0.1,  0.0],
            [+0.2,  0.0],
            [+0.5,  0.1],
            [+1.0,  0.2],
        ]),
        LeafletThickness=0.01,  # relative to leaf width
        LeafletBend=1.0,
        LeafletBendRelStdDev=0.1,
        LeafletBendXFuncMethod='LeafBendX',
        # Bud
        BudAgeMature=4,
        BudLength=0.5,
        FlowerAgeMature=4,
        PistilWidth=0.1,
        PistilLength=0.7,
        # Petal
        PetalWMax=5,
        PetalLength=1.0,
        PetalWidth=0.3,
        PetalWidthXFunc='linear',
        PetalWidthXFuncSlope=-1.0,
        PetalWidthXFuncIntercept=1.0,
        # Sepal
        SepalWMax=5,
        SepalLength=0.5,
        SepalWidth=0.5,
        SepalWidthXFunc='linear',
        SepalWidthXFuncSlope=-1.0,
        SepalWidthXFuncIntercept=1.0,
        # Fruit cluster
        FruitClusterAgeMature=3,
        FruitClusterBpMax=1,  # This is the number of branches/cluster
        FruitClusterNpMax=3,  # This is the number of nodes/cluster
        FruitClusterWMax=1,
        FruitClusterNFirst=1,
        FruitClusterNPeriod=2,
        FruitClusterAngle=70,
        FruitClusterRotationAngle=0,
        FruitClusterInternodeLength=0.75,
        # FruitingComplex
        FruitingComplexWMax=1,
        FruitingComplexNpFirst=0,
        FruitingComplexNpPeriod=1,
        FruitingComplexAngle=70,
        FruitingComplexAngleNpFunc='expression',
        FruitingComplexAngleNpFuncExpression=(
            '0.0 if Np >= 5 else 1.0'
        ),
        FruitingComplexRotationAngle=90,
        FruitingComplexRotationAngleNpFunc='expression',
        FruitingComplexRotationAngleNpFuncExpression=(
            '1.0 if (Np % 2) == 0 else 3.0'
        ),
        FruitingComplexTiltAngle=90,
        FruitingComplexTiltAngleNpFunc='expression',
        FruitingComplexTiltAngleNpFuncExpression=(
            '3.0 if (Np % 2) == 0 else 1.0'
        ),
        # Fruit
        FruitAgeMature=3,
        FruitLength=1.0,
        FruitLengthRelStdDev=0.01,
        PedicelLength=0.5,
        PedicelBend=80.0,
        FruitColor='RdYlGn_r',
    )
