import pytest
import os
import copy
from canopy_factory.utils import get_class_registry, DataProcessor


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


def pytest_generate_tests(metafunc):
    if not (metafunc.cls and metafunc.cls._registry_name):
        return
    params = copy.deepcopy(metafunc.cls._params)
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
    if param_unset('id'):
        assert 'crop' in metafunc.fixturenames and 'crop' in params
        crops = params.pop('crop')
        argvalues = []
        argvalues = []
        for k in crops:
            ids = DataProcessor.available_ids(k)
            if ids:
                for vv in params.get('id', ids):
                    argvalues.append((k, vv))
            else:
                argvalues.append((k, None))
        params['crop'] = {
            'names': ['crop', 'id'],
            'values': argvalues,
            'ids': [f'{k}-{v}' for k, v in argvalues],
            'scope': 'class',
        }
    for k, v in params.items():
        scope = "function" if k in ['output'] else "class"
        if isinstance(v, list):
            metafunc.parametrize(k, v, scope=scope)
        elif isinstance(v, dict):
            names = v.pop('names', k)
            values = v.pop('values')
            v.setdefault("scope", scope)
            metafunc.parametrize(names, values, **v)
        else:
            raise TypeError(type(v))


@pytest.fixture(scope="session")
def test_output_dir():
    r"""str: Output directory containing expected output for tests."""
    return os.path.join(os.path.dirname(__file__), 'data')
