import pytest
import os
import copy
import numpy as np
from canopy_factory.utils import (
    get_class_registry, DataProcessor, units, cfg
)


_param_args = ['crop', 'id', 'data_year']
_step_outputs = []


class NestedAssertionError(AssertionError):

    def __init__(self, nested):
        self.nested = nested
        msg = ''
        for k, v in nested.items():
            msg += f'\n\n{k}\n\t' + v.replace('\n', '\n\t')
        super(NestedAssertionError, self).__init__(msg)


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
    parser.addoption(
        "--preserve-step-output",
        help="Preserve step output to speed up consequent runs.",
        action='store_true',
    )
    parser.addoption(
        "--ignore-data",
        help="Run tests as if the collected data does not exist.",
        action='store_true',
    )
    for k in _param_args:
        parser.addoption("--" + k.replace('_', '-'), type=str)


def pytest_runtest_setup(item):
    if not (item.cls and item.cls._registry_name
            and (item.config.getoption('task')
                 or item.config.getoption('skip_task')
                 or item.config.getoption('ignore_data')
                 or not os.path.isdir(cfg['directories']['input']))):
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
    if ((item.config.getoption('ignore_data')
         or not os.path.isdir(cfg['directories']['input']))):
        vid = item.callspec.params['arguments'].get(
            'id', item.callspec.params.get('id', 'default'))
        if vid != 'default':
            pytest.skip(f"No local data for id=\"{vid}\"")
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
            default_param = {'id': 'default'}
            if metafunc.config.getoption('ignore_data'):
                DataProcessor._ignore_data = True
                kparam = []
            else:
                kparam = DataProcessor.available_param(
                    param_names, crop=k, **param_kwargs)
            if kparam:
                argvalues += kparam
            argvalues.append(
                tuple([k] + [default_param.get(kp, None)
                             for kp in implicit_param]))
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
def add_step_output():
    r"""Add a step output that should be cleaned up at the end of the
    session.

    Args:
        x (str, list): One or more step outputs to cleanup.

    """

    def _add_step_output(x):
        if not isinstance(x, list):
            x = [x]
        for xx in x:
            if os.path.isfile(xx):  # Prevent removing existing files
                continue
            _step_outputs.append(xx)

    return _add_step_output


@pytest.fixture(scope="session", autouse=True)
def cleanup_step_files(request):
    r"""Cleanup any step outputs that could be reused between tests."""
    yield
    if not request.config.getoption("--preserve-step-output"):
        for x in _step_outputs:
            if os.path.isfile(x):
                os.remove(x)


@pytest.fixture(scope="session")
def assert_allclose():
    r"""Assert that arrays are close."""

    def _assert_allclose(a, b, rtol=1e-07, atol=1e-15, **kwargs):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)

    return _assert_allclose


@pytest.fixture(scope="session")
def assert_nested_allclose(assert_allclose):
    from collections import OrderedDict

    def _assert_nested_allclose(a, b, **kwargs):
        errors = {}
        if isinstance(b, (list, tuple)):
            assert isinstance(a, type(b))
            assert len(a) == len(b)
            for i, (ia, ib) in enumerate(zip(a, b)):
                try:
                    _assert_nested_allclose(ia, ib, **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{i}->{kerr}'] = verr
                    else:
                        errors[f'{i}'] = e.args[0]
        elif isinstance(b, (dict, OrderedDict)):
            assert isinstance(a, type(b))
            assert len(a) == len(b)
            a_keys = list(sorted(a.keys()))
            b_keys = list(sorted(b.keys()))
            assert a_keys == b_keys
            for k in b_keys:
                try:
                    _assert_nested_allclose(a[k], b[k], **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{k}->{kerr}'] = verr
                    else:
                        errors[k] = e.args[0]
        elif isinstance(b, (units.Quantity, units.QuantityArray)):
            assert isinstance(a, type(b))
            assert a.units == b.units
            assert_allclose(a.value, b.value, **kwargs)
        elif isinstance(b, np.ndarray):
            assert_allclose(a, b, **kwargs)
        else:
            assert a == b
        if errors:
            raise NestedAssertionError(errors)

    return _assert_nested_allclose


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


@pytest.fixture(scope="session")
def compare_approx_csv(compare_approx):
    r"""Compare two objects using approximate equality for floats.

    Args:
        actual (object): Actual object.
        expected (object): Expected object.

    """

    def _compare_approx_csv(actual, expected, **kwargs):
        flag = 'HEADER_JSON'
        if flag in expected:
            assert flag in actual
            expected_keys = list(expected[flag].keys())
            actual_keys = list(actual[flag].keys())
            assert actual_keys == expected_keys
            expected = {k: v for k, v in expected.items() if k != flag}
            actual = {k: v for k, v in actual.items() if k != flag}
        compare_approx(actual, expected, **kwargs)

    return _compare_approx_csv


@pytest.fixture(scope="session")
def compare_approx(assert_nested_allclose):
    r"""Compare two objects using approximate equality for floats.

    Args:
        actual (object): Actual object.
        expected (object): Expected object.

    """
    # def _compare_approx(actual, expected, nan_ok=True, **kwargs):
    #     assert_equal_approx(actual, expected, nan_ok=nan_ok, **kwargs)
    return assert_nested_allclose


@pytest.fixture(scope="session")
def compare_bytes_csv(compare_bytes):
    r"""Compare CSV bytes in chunks after removing the JSON header.

    Args:
        actual (bytes): Actual bytes.
        expected (bytes): Expected bytes.

    """
    def _compare_bytes_csv(actual, expected):
        flag = b'# HEADER_JSON: '
        if flag in expected:
            assert flag in actual
            idx = actual.index(flag)
            actual = actual[idx:].split(b'\n', maxsplit=1)[-1]
            idx = expected.index(flag)
            expected = expected[idx:].split(b'\n', maxsplit=1)[-1]
        compare_bytes(actual, expected)

    return _compare_bytes_csv


@pytest.fixture(scope="session")
def compare_bytes():
    r"""Compare bytes in chunks.

    Args:
        actual (bytes): Actual bytes.
        expected (bytes): Expected bytes.

    """

    def _compare_bytes(actual, expected):
        chunk_size = 1000
        len_actual = len(actual)
        len_expected = len(expected)
        pos = 0
        maxpos = max([len_actual, len_expected])
        while pos < maxpos:
            pos_act = min(pos, len_actual)
            pos_exp = min(pos, len_expected)
            chunk_act = actual[
                pos_act:min(pos_act + chunk_size, len_actual)]
            chunk_exp = expected[
                pos_exp:min(pos_exp + chunk_size, len_expected)]
            assert chunk_act == chunk_exp
            pos += chunk_size
        assert len_actual == len_expected
        assert actual == expected

    return _compare_bytes
