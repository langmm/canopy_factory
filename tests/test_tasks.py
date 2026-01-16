import pytest
import os
import uuid
import shutil
from canopy_factory import utils
# TODO
# - test for id='all'
# - check tests/data location for output within package to prevent
#   duplication during local runs
# - unit tests


class TestTask(object, metaclass=utils.RegisteredMetaClass):
    r"""Class for testing task output."""

    _registry_key = 'task_test'
    _registry_name = None
    _params = {
        'crop': ['maize'],
        'id': ['B73_WT'],
        'data_year': ['2024'],
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
    def instance(self, task_class, instance_kwargs, test_output_dir):
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

    def test_output(self, instance, output, compare_method,
                    fname_actual, fname_expected, approx_nested,
                    create_missing_data, overwrite_existing_data,
                    compare_approx, compare_bytes,
                    compare_approx_csv, compare_bytes_csv):
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
        eval(f'compare_{compare_method}')(data_actual, data_expected)

    def prepare_comparison_data(self, output, actual, expected):
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
    _params = {}  # Perform for all crops/ids


class TestGenerateTask(TestTask):
    r"""Class for testing generate."""

    _registry_name = 'generate'
    _params = {
        'data_year': ['2024'],
        'arguments': {
            'values': [
                {'canopy': 'single'},
                {'crop': 'maize', 'id': 'B73_WT', 'canopy': 'unique'},
                {'crop': 'maize', 'id': 'B73_WT', 'canopy': 'tile'},
            ],
        },
    }

    @pytest.fixture
    def compare_method(self, output, arguments):
        r"""str: Method that should be used to compare output files."""
        if arguments.get('canopy', 'single') in ['unique', 'tile']:
            return False  # Large
        return self._compare_methods.get(output, 'bytes')


class TestRayTraceTask(TestTask):
    r"""Class for testing raytrace."""

    _registry_name = 'raytrace'
    _params = {
        'arguments': {
            'values': [
                {'crop': 'maize', 'id': 'B73_WT', 'data_year': '2024',
                 'canopy': 'single'},
                {'crop': 'maize', 'id': 'B73_WT', 'data_year': '2024',
                 'canopy': 'virtual', 'periodic_canopy': True},
                {'crop': 'maize', 'id': 'B73_WT', 'data_year': '2024',
                 'canopy': 'virtual_single'},
            ],
        },
    }
    _compare_methods = {
        'raytrace': 'bytes_csv',
        'raytrace_limits': 'approx',
        'traced_mesh': False,
    }


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
                {'crop': 'maize', 'id': 'B73_WT', 'data_year': '2024',
                 'canopy': 'virtual',
                 'periodic_canopy': True,
                 'duration': '2 hr'},
                {'crop': 'maize', 'id': 'B73_WT', 'data_year': '2024',
                 'canopy': 'virtual_single',
                 'duration': '2 hr'},
            ],
        },
    }
    _compare_methods = {
        'totals': 'approx',
    }

    def prepare_comparison_data(self, output, actual, expected):
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
            assert 'compute_time' in expected
            assert 'compute_time' in actual
            expected = {k: v for k, v in expected.items()
                        if k != 'compute_time'}
            actual = {k: v for k, v in actual.items()
                      if k != 'compute_time'}
        return super(TestTotalsTask, self).prepare_comparison_data(
            output, actual, expected)


class TestAnimateTask(TestTask):
    r"""Class for testing animate."""

    _registry_name = 'animate'
    _params = TestTotalsTask._params


# class TestMatchQueryTask(TestTask):
#     r"""Class for testing match query."""

#     _registry_name = 'match_query'
#     _params = {
#         'crop': ['maize'],
#         'id': ['rdla'],
#         'data_year': ['2024'],
#         'arguments': {
#             'values': [
#                 {'canopy': 'virtual', 'periodic_canopy': True},
#             ],
#         },
#     }


# class TestPhotosynthesisTask(TestTask):
#     r"""Class for testing photosynthesis calc."""

#     _registry_name = 'photosynthesis'
