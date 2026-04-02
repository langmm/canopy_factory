import copy
import pprint
import numpy as np
from canopy_factory.crops.base import CurvePlantParameter, PlantGenerator


class MonocotGenerator(PlantGenerator):
    r"""Class for generating generic monocot plant geometries."""

    _plant_name = 'monocot'
    _components = dict(
        PlantGenerator._components,
        Leaf={
            'defaults': {
                'WMax': 1,
                'NFirst': 0,  # No separate profile for cotyledons
            },
        },
    )
    _defaults = dict(
        PlantGenerator._defaults,
        NMax=10,
        NodeElements=['Cotyledon', 'Leaf'],
        LeafAngle=30,
        LeafMethod='sweep',
        LeafLength=60.0,
        LeafLengthRelStdDev=0.2,
        LeafWidthFunc='pow',
        LeafWidthFuncVarName='LeafLength',
        LeafWidthFuncExp=0.25,
        LeafWidthXFunc='linear',
        LeafWidthXFuncSlope=-1.0,
        LeafWidthXFuncIntercept=1.0,
        LeafThickness=0.1,  # relative to leaf width
        InternodeLength=0.0,
        InternodeWidth=1.0,
        InternodeWidthNFunc='linear',
        InternodeWidthNFuncSlope=-1.0,
        InternodeWidthNFuncIntercept=1.0,
        InternodeRotationAngle=180,
        InternodeRotationAngleRelStdDev=0.1,
    )

    def __init__(self, **kwargs):
        super(MonocotGenerator, self).__init__(**kwargs)
        if self.unfurl_leaves:
            self.update_curve_param(
                'LeafProfile', 'circle',
                patch_param={'VarName': 'X'},
            )

    @property
    def unfurl_leaves(self):
        r"""bool: True if LeafUnfurled set."""
        return self.get('LeafUnfurled', False)

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
