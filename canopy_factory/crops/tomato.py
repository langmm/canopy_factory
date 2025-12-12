from canopy_factory.crops.dicot import DicotGenerator


class TomatoGenerator(DicotGenerator):
    r"""Class for generating tomato plant geometries."""

    _plant_name = 'tomato'
    _components = dict(
        DicotGenerator._components,
        Pedicel={},
        Fruit={
            'defaults': {
                'NFirst': 2,
                'NPeriod': 2,
            },
        },
    )
    _defaults = dict(
        DicotGenerator._defaults,
        NMax=15,
        # Pedicel parameters
        PedicelLength=1,
        PedicelWidth=0.125,
        # Fruit parameters
        FruitLength=1,
        FruitWMax=2,
        FruitAngle=100,
        # FruitRotationAngle=90,
        FruitRotationAngle=360,
        FruitRotationAngleWFunc='linear',
        FruitRotationAngleWFuncSlope=0.75,
        FruitRotationAngleWFuncIntercept=0.25,
    )
