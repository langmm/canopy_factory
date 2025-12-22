import pytest
import os
import copy
from canopy_factory.utils import get_class_registry, DataProcessor, units


_param_args = ['crop', 'id', 'data_year']


def pytest_addoption(parser):
    parser.addoption(
        "--task", choices=list(get_class_registry().keys('task')),
        help="Only run tests for the named task(s)",
        nargs='*',
    )
    parser.addoption(
        "--skip-task", choices=list(get_class_registry().keys('task')),
        help="Skip tests for the named task(s)",
        nargs='*',
    )
    parser.addoption(
        "--task-output",
        help="Only run tests for the named task output(s)",
        nargs='*',
    )
    parser.addoption(
        "--create-missing-data",
        help="Create missing test data.",
        action='store_true',
    )
    parser.addoption(
        "--overwrite-existing-data",
        help="Overwrite the existing test data.",
        action='store_true',
    )
    for k in _param_args:
        parser.addoption("--" + k.replace('_', '-'), type=str)


def pytest_runtest_setup(item):
    if not (item.cls and item.cls._registry_name
            and (item.config.getoption('task')
                 or item.config.getoption('skip_task'))):
        return
    if ((item.config.getoption('task')
         and (item.cls._registry_name not in
              item.config.getoption('task')))):
        pytest.skip(f"\"{item.cls._registry_name}\" "
                    f"task tests not enabled")
    if ((item.config.getoption('skip_task')
         and (item.cls._registry_name in
              item.config.getoption('skip_task')))):
        pytest.skip(f"\"{item.cls._registry_name}\" "
                    f"task tests disabled")
    for k in _param_args:
        v = item.config.getoption(k)
        if not v:
            continue
        vparam = item.callspec.params['arguments'].get(
            k, item.callspec.params[k])
        if vparam != v:
            pytest.skip(f"{k} = {vparam} not selected")


def pytest_generate_tests(metafunc):
    if not (metafunc.cls and metafunc.cls._registry_name):
        return
    params = copy.deepcopy(metafunc.cls._params)
    params_data = [
        x for x in metafunc.cls._params_data if x != 'crop'
    ]
    cls = get_class_registry().get('task', metafunc.cls._registry_name)

    def param_unset(k):
        return (k in metafunc.fixturenames and k not in params)

    if param_unset('output'):
        params['output'] = cls._outputs_local
    if ((metafunc.config.getoption('task_output')
         and 'output' in metafunc.fixturenames)):
        params['output'] = [
            k for k in params['output']
            if k in metafunc.config.getoption('task_output')
        ]
    if param_unset('arguments'):
        params['arguments'] = {
            'values': [{}],
            'ids': ['']
        }
    if param_unset('crop'):
        params['crop'] = list(get_class_registry().keys('crop'))
    implicit_param = [x for x in params_data if param_unset(x)]
    if implicit_param:
        assert 'crop' in metafunc.fixturenames and 'crop' in params
        crops = params.pop('crop')
        argnames = ['crop'] + implicit_param
        param_map = {'data_year': 'year'}
        param_names = [param_map.get(x, x) for x in argnames]
        param_kwargs = {
            param_map.get(x, x): params.get(x, None)
            for x in params_data
        }
        argvalues = []
        for k in crops:
            kparam = DataProcessor.available_param(
                param_names, crop=k, **param_kwargs)
            if kparam:
                argvalues += kparam
            else:
                argvalues.append(
                    tuple([k] + [None for _ in implicit_param]))
        params['crop'] = {
            'names': argnames,
            'values': argvalues,
            'ids': ['-'.join([str(v) for v in x]) for x in argvalues],
            'scope': 'class',
        }
    order = ['crop'] + params_data + ['arguments', 'output']
    order_unsorted = []
    for k in list(params.keys()):
        v = params[k]
        scope = "function" if k in ['output'] else "class"
        if isinstance(v, list):
            args = (k, v)
            kwargs = {'scope': scope}
        elif isinstance(v, dict):
            names = v.pop('names', k)
            values = v.pop('values')
            args = (names, values)
            kwargs = v
            kwargs.setdefault("scope", scope)
        else:
            raise TypeError(type(v))
        params[k] = (args, kwargs)
        if k not in order:
            order_unsorted.append(k)

    def param_key(k):
        hierarchy = ['session', 'package', 'module', 'class', 'function']
        scope = params[k][1]['scope']
        return hierarchy.index(scope)

    order += sorted(order_unsorted, key=param_key)
    for k in order:
        if k not in params:
            continue
        args, kwargs = params[k][:]
        metafunc.parametrize(*args, **kwargs)


@pytest.fixture(scope="session")
def test_output_dir():
    r"""str: Output directory containing expected output for tests."""
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope="session")
def create_missing_data(request):
    return request.config.getoption("--create-missing-data")


@pytest.fixture(scope="session")
def overwrite_existing_data(request):
    return request.config.getoption("--overwrite-existing-data")


@pytest.fixture(scope="session")
def patch_equality():
    def patch_equality_w(obj, method):
        class EqualityWrapper:
            def __init__(self, x):
                self.x = x

            def __str__(self):
                return f"EqualityWrapper({self.x!s})"

            def __repr__(self):
                return f"EqualityWrapper({self.x!r})"

            def __eq__(self, other):
                if isinstance(other, EqualityWrapper):
                    y = other.x
                else:
                    y = other
                if not isinstance(y, self.x.__class__):
                    return False
                return method(self.x, y)
        return EqualityWrapper(obj)
    return patch_equality_w


@pytest.fixture(scope="session")
def apply_nested():
    r"""callable: Method to apply to elements in a nested data
    structure."""
    from collections import OrderedDict

    def _apply_nested(expected, method, skip_types=None, **kwargs):
        if isinstance(expected, (list, tuple)):
            if skip_types:
                kwargs['skip_types'] = skip_types
            return type(expected)(
                [_apply_nested(x, method, **kwargs) for x in expected])
        elif isinstance(expected, (dict, OrderedDict)):
            if skip_types:
                kwargs['skip_types'] = skip_types
            return type(expected)(
                [(k, _apply_nested(v, method, **kwargs))
                 for k, v in expected.items()])
        elif skip_types and isinstance(expected, skip_types):
            return expected
        else:
            return method(expected, **kwargs)

    return _apply_nested


@pytest.fixture(scope="session")
def assert_equal_approx(approx_nested, patch_equalities):
    r"""callable: Method to assert that two structures are equivalent."""

    def _assert_equal_approx(lhs, rhs, **kwargs):
        assert (
            patch_equalities(lhs, **kwargs)
            == approx_nested(rhs, **kwargs)
        )

    return _assert_equal_approx


@pytest.fixture(scope="session")
def patch_equalities(apply_nested, patch_equality):
    r"""callable: Method to patch equalities for approx comparison."""

    def _patch_equality(expected, **kwargs):
        if isinstance(expected, (units.Quantity, units.QuantityArray)):
            def units_equality(a, b):
                if a.units != b.units:
                    return False
                return a.value == pytest.approx(b.value, **kwargs)
            return patch_equality(expected, units_equality)
        return expected

    def _patch_equalities(expected, **kwargs):
        return apply_nested(expected, _patch_equality, **kwargs)

    return _patch_equalities


@pytest.fixture(scope="session")
def approx_nested(apply_nested):
    r"""callable: Method to check that a nested data structure is
    approximately equal to another."""

    def _approx_nested(expected, **kwargs):
        return apply_nested(
            expected, pytest.approx,
            skip_types=(units.Quantity, units.QuantityArray),
            **kwargs
        )

    return _approx_nested
