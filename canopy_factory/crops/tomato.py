import yggdrasil_rapidjson as rapidjson
from canopy_factory.crops.dicot import DicotGenerator


class TomatoGenerator(DicotGenerator):
    r"""Class for generating tomato plant geometries."""

    _plant_name = 'tomato'
    _defaults = dict(
        DicotGenerator._defaults,
        NMax=10,
        ApexBLast=1,
        # Plastocron=rapidjson.units.Quantity(1.0, 'days'),
        # Pedicel parameters
        PedicelLength=0.5,
        PedicelWidth=0.125,
        # Fruit parameters
        FruitNFirst=1,
        FruitNPeriod=2,
        FruitWMax=2,
        FruitLength=1.0,
        FruitAngle=100,
        FruitAgeMature=rapidjson.units.Quantity(10.0, 'days'),
        FruitColor='RdYlGn_r',
        FruitRotationAngle=360,
        FruitRotationAngleWFunc='linear',
        FruitRotationAngleWFuncSlope=1.0,
        FruitRotationAngleWFuncIntercept=0.25,
        InternodeLength=8.0,
        InternodeAgeMature=3,
        LeafAgeMature=6,
        LeafWMax=1,
        ApexAngleBFunc='expression',
        ApexAngleBFuncExpression='1.0 if B > 0 else 0.25',
    )
