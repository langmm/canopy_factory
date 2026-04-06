import pytest
import os
import uuid
import shutil
from canopy_factory import utils
from canopy_factory.cli import IterationTaskBase
# TODO
# - test for id='all'
# - check tests/data location for output within package to prevent
#   duplication during local runs
# - unit tests


def test_nested(assert_nested_allclose):
    import copy
    import numpy as np
    from yggdrasil_rapidjson import units
    a = {
        0: np.ones((5, )),
        1: [np.zeros((2, )), np.ones((3, ))],
        2: units.QuantityArray(np.arange(5), "cm"),
    }

    def create_copy(value):
        out = copy.deepcopy(a)
        out[0][1] += value
        out[1][0][1] += value
        out[1][1][1] += value
        out[2][1] = out[2][1] + units.Quantity(value, "cm")
        return out

    assert_nested_allclose(a, a)
    b = create_copy(1e-16)
    assert_nested_allclose(a, b)
    # c = create_copy(266)
    # assert_nested_allclose(a, c)


class TestTask(object, metaclass=utils.RegisteredMetaClass):
    r"""Class for testing task output."""

    _registry_key = 'task_test'
    _registry_name = None
    _params = {
        'crop': ['maize'],
        'id': ['default'],
        # 'data_year': ['2024'],
    }
    _params_data = ['id', 'data_year']
    _compare_methods = {}

    @pytest.fixture(scope="class")
    def task_name(self):
        r"""str: Name of the task being tested."""
        if self._registry_name is None:
            pytest.skip()
        return self._registry_name

    @pytest.fixture(scope="class")
    def task_class(self, task_name):
        r"""type: Task class."""
        return utils.get_class_registry().get('task', task_name)

    @pytest.fixture(scope="class")
    def crop_class(self, crop):
        r"""type: Crop generator class."""
        return utils.get_class_registry().get('crop', crop)

    @pytest.fixture
    def compare_method(self, output):
        r"""str: Method that should be used to compare output files."""
        return self._compare_methods.get(output, 'bytes')

    @pytest.fixture(scope="class")
    def instance_kwargs(self, task_class, crop_class,
                        crop, id, data_year, arguments, test_output_dir):
        r"""dict: Keyword arguments for creating the instance."""
        out = {'output_dir': test_output_dir}
        for k in ['crop'] + self._params_data:
            x = eval(k)
            if arguments.get(k, x) != x:
                pytest.skip(f'{k} specified by arguments')
            out[k] = x
        for k in task_class._outputs_local:
            out[f'output_{k}'] = True
        out.update(**arguments)
        return out

    @pytest.fixture(scope="class")
    def instance(self, task_class, instance_kwargs, test_output_dir,
                 add_step_output):
        r"""TaskBase: Task instance for testing."""
        out = task_class.from_kwargs(instance_kwargs)
        out._testing_default_files = {}
        testid = str(uuid.uuid4())
        outputs_remove = out._outputs_local
        for k in out._outputs_total:
            kout = out.output_argument(k)
            kpath = kout.path
            if kpath is None:
                kpath = out.output_file(k, return_disabled=True)
            assert isinstance(kpath, str)
            out._testing_default_files[k] = kpath
            if k in outputs_remove:
                kout._generated_path = testid.join(os.path.splitext(
                    kpath))
        if isinstance(out, IterationTaskBase):
            for iargs_overwrite in out.step_args_full():
                iargs = out._step_task.copy_external_args(
                    out.args, initialize=True,
                    args_overwrite=iargs_overwrite,
                )
                for k in out._step_task._outputs_local:
                    getattr(iargs, f'output_{k}')._copied_args = iargs
                    add_step_output(getattr(iargs, f'output_{k}').path)
        try:
            yield out
        finally:
            for k in outputs_remove:
                kout = out.output_file(k)
                if os.path.isfile(kout):
                    os.remove(kout)

    @pytest.fixture
    def fname_actual(self, instance, output):
        r"""str: Path of file containing generated result."""
        return instance.output_file(output)

    @pytest.fixture
    def fname_expected(self, instance, output):
        r"""str: Path of file containing expected result."""
        return instance._testing_default_files[output]

    @pytest.fixture(scope="class")
    def tolerance_approx(self):
        r"""Method to determine tolerance to use for approximate
        comparison based on the type of output."""

        def _tolerance_approx(output):
            if output in ['raytrace', 'raytrace_stats', 'totals',
                          'render_camera']:
                return {'rtol': 1e-6}
            return {}

        return _tolerance_approx

    def test_output(self, instance, output, compare_method,
                    fname_actual, fname_expected, approx_nested,
                    create_missing_data, overwrite_existing_data,
                    compare_approx, compare_bytes,
                    compare_approx_csv, compare_bytes_csv,
                    tolerance_approx):
        r"""Test creating output."""
        if fname_expected.endswith(('.png', '.gif')):
            # Don't compare png binaries
            compare_method = False
        if not (compare_method is False or create_missing_data
                or os.path.isfile(fname_expected)):
            raise AssertionError(f'Expected \"{output}\" output '
                                 f'does not exist: {fname_expected}')
        data_actual = instance.get_output(output)
        assert os.path.isfile(fname_actual)
        if compare_method is False:
            return
        if (((create_missing_data and not os.path.isfile(fname_expected))
             or overwrite_existing_data)):
            print(f"Creating test file: {fname_expected}")
            shutil.copyfile(fname_actual, fname_expected)
        if compare_method.startswith('bytes'):
            with open(fname_actual, 'rb') as fd:
                data_actual = fd.read()
            with open(fname_expected, 'rb') as fd:
                data_expected = fd.read()
        else:
            data_expected = instance.read_output(output, fname_expected)
        data_actual, data_expected = self.prepare_comparison_data(
            output, data_actual, data_expected)
        kws = (
            {} if ('approx' not in compare_method)
            else tolerance_approx(output)
        )
        eval(f'compare_{compare_method}')(data_actual, data_expected, **kws)

    @classmethod
    def prepare_comparison_data(cls, output, actual, expected):
        r"""Perform an actions necessary to modify the data prior to
        comparison.

        Args:
            output (str): Type of output being compared.
            actual (object): Actual object.
            expected (object): Expected object.

        Returns:
            tuple: Update actual & expected objects for comparison.

        """
        return actual, expected


class TestLayoutTask(TestTask):
    r"""Class for testing layout."""

    _registry_name = 'layout'
    _params = {
        'arguments': {
            'values': [
                {'canopy': 'single'},
                {'canopy': 'single', 'periodic_canopy': True},
                {'canopy': 'unique', 'periodic_canopy': True},
                {'canopy': 'unique', 'time': 'noon'},
                {'canopy': 'tiled'},
                {'canopy': 'virtual'},
                {'canopy': 'virtual', 'periodic_canopy': True},
            ],
            'ids': [
                'single',
                'single_periodic',
                'unique',
                'unique_periodic',
                'tiled',
                'virtual',
                'virtual_periodic',
            ],
        },
    }

    @pytest.fixture(scope="class")
    def instance_kwargs(self, task_class, arguments, test_output_dir):
        r"""dict: Keyword arguments for creating the instance."""
        out = dict(
            arguments,
            output_dir=test_output_dir,
        )
        for k in task_class._outputs_local:
            out[f'output_{k}'] = True
        return out


class TestParametrizeCropTask(TestTask):
    r"""Class for testing parametrization."""

    _registry_name = 'parametrize'
    _params = {  # Perform for all crops/ids
        'arguments': {
            'values': [
                {},
                {'crop': 'maize', 'id': 'B73_WT',
                 'piecewise_param': 'N', 'data_year': '2024'},
            ],
        },
    }


class TestGenerateTask(TestTask):
    r"""Class for testing generate."""

    _registry_name = 'generate'
    _params = {
        'arguments': {
            'values': [
                {'canopy': 'single', 'data_year': '2024'},
                {'crop': 'maize', 'id': 'default', 'canopy': 'unique'},
                {'crop': 'maize', 'id': 'default', 'canopy': 'tile'},
                {'crop': 'maize', 'id': 'B73_WT', 'canopy': 'single',
                 'piecewise_param': 'N', 'data_year': '2024'},
            ],
        },
    }
    # _compare_methods = {'generate': 'approx'}

    @pytest.fixture
    def compare_method(self, output, arguments):
        r"""str: Method that should be used to compare output files."""
        if arguments.get('canopy', 'single') in ['unique', 'tile']:
            return False  # Large
        return self._compare_methods.get(output, 'bytes')

    @classmethod
    def prepare_comparison_data(cls, output, actual, expected):
        r"""Perform an actions necessary to modify the data prior to
        comparison.

        Args:
            output (str): Type of output being compared.
            actual (object): Actual object.
            expected (object): Expected object.

        Returns:
            tuple: Update actual & expected objects for comparison.

        """
        if output == 'generate' and not isinstance(expected, (str, bytes)):
            expected = utils.get_mesh_dict(expected)
            actual = utils.get_mesh_dict(actual)
        return actual, expected


class TestRayTraceTask(TestTask):
    r"""Class for testing raytrace."""

    _registry_name = 'raytrace'
    _params = {
        'arguments': {
            'values': [
                {'crop': 'maize', 'id': 'default', 'canopy': 'single'},
                {'crop': 'maize', 'id': 'default', 'canopy': 'virtual',
                 'periodic_canopy': True},
                {'crop': 'maize', 'id': 'default',
                 'canopy': 'virtual_single'},
                {'crop': 'maize', 'id': 'default', 'canopy': 'virtual',
                 'periodic_canopy': True,
                 'time': '2024-06-20 13:54:43.950993-05:00'},
            ],
            'ids': [
                'single',
                'virtual_periodic',
                'virtual_single',
                'virtual_periodic_t2',
            ],
        },
    }
    _compare_methods = {
        # 'raytrace': 'bytes_csv',
        'raytrace': 'approx_csv',
        'raytrace_limits': 'approx',
        'raytrace_stats': 'approx',
        'traced_mesh': False,
    }

    @classmethod
    def prepare_comparison_data(cls, output, actual, expected):
        r"""Perform an actions necessary to modify the data prior to
        comparison.

        Args:
            output (str): Type of output being compared.
            actual (object): Actual object.
            expected (object): Expected object.

        Returns:
            tuple: Update actual & expected objects for comparison.

        """
        if output == 'raytrace_stats' or (output == 'raytrace'
                                          and isinstance(expected, dict)):
            if output == 'raytrace':
                assert 'HEADER_JSON' in expected
                assert 'HEADER_JSON' in actual
                expected_header = expected['HEADER_JSON']
                actual_header = actual['HEADER_JSON']
            else:
                expected_header = expected
                actual_header = actual
            assert 'compute_time' in expected_header
            assert 'compute_time' in actual_header
            expected_header = {k: v for k, v in expected_header.items()
                               if k != 'compute_time'}
            actual_header = {k: v for k, v in actual_header.items()
                             if k != 'compute_time'}
            if output == 'raytrace':
                expected = {k: v for k, v in expected.items()}
                actual = {k: v for k, v in actual.items()}
                expected['HEADER_JSON'] = expected_header
                actual['HEADER_JSON'] = actual_header
            else:
                expected = expected_header
                actual = actual_header
        return super(TestRayTraceTask, cls).prepare_comparison_data(
            output, actual, expected)


class TestRenderTask(TestTask):
    r"""Class for testing render."""

    _registry_name = 'render'
    _params = TestRayTraceTask._params
    _compare_methods = {
        'render_camera': 'approx',
    }


class TestTotalsTask(TestTask):
    r"""Class for testing totals."""

    _registry_name = 'totals'
    _params = {
        'arguments': {
            'values': [
                dict(x, start_time='noon', duration='1 hr')
                for x in TestRayTraceTask._params['arguments']['values']
                if x.get('canopy', None) != 'single' and 'time' not in x
            ],
        },
    }
    _compare_methods = {
        'totals': 'approx',
    }

    @classmethod
    def prepare_comparison_data(cls, output, actual, expected):
        r"""Perform an actions necessary to modify the data prior to
        comparison.

        Args:
            output (str): Type of output being compared.
            actual (object): Actual object.
            expected (object): Expected object.

        Returns:
            tuple: Update actual & expected objects for comparison.

        """
        if output == 'totals':
            return TestRayTraceTask.prepare_comparison_data(
                'raytrace_stats', actual, expected)
        return super(TestTotalsTask, cls).prepare_comparison_data(
            output, actual, expected)


class TestAnimateTask(TestTask):
    r"""Class for testing animate."""

    _registry_name = 'animate'
    _params = {
        'arguments': {
            'values': [
                TestTotalsTask._params['arguments']['values'][0],
                dict(
                    inset_totals=True,
                    **TestTotalsTask._params['arguments']['values'][0]
                ),
            ],
        },
    }


class TestMatchQueryTask(TestTask):
    r"""Class for testing match query."""

    _registry_name = 'match_query'
    _params = {
        'crop': ['maize'],
        'id': ['B73_rdla'],
        'data_year': ['2024'],
        'arguments': {
            'values': [
                {'canopy': 'virtual_single',
                 'vary': 'row_spacing',
                 'tolerance': 0.1,
                 'initial_value': 63.510483327203126,
                 'dont_write_raytrace': True,
                 'dont_write_raytrace_stats': True},
            ],
        },
    }
    _compare_methods = {
        'match_query': 'approx',
    }


# class TestPhotosynthesisTask(TestTask):
#     r"""Class for testing photosynthesis calc."""

#     _registry_name = 'photosynthesis'
