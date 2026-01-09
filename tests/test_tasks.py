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
    }
    _disable_compare = False

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

    @pytest.fixture(scope="class")
    def disable_compare(self):
        r"""bool: True if byte-wise comparison should be disabled."""
        return self._disable_compare

    @pytest.fixture(scope="class")
    def instance_kwargs(self, task_class, crop_class,
                        crop, id, arguments, test_output_dir):
        r"""dict: Keyword arguments for creating the instance."""
        if arguments.get('id', id) != id:
            pytest.skip('id specified by arguments')
        if arguments.get('crop', crop) != crop:
            pytest.skip('crop specified by arguments')
        out = {'crop': crop, 'id': id, 'output_dir': test_output_dir}
        for k in task_class._outputs_local:
            out[f'output_{k}'] = True
        out.update(**arguments)
        # if crop_class._default_data and 'data' not in out:
        #     out['data'] = crop_class._default_data
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

    def test_output(self, instance, output, disable_compare,
                    fname_actual, fname_expected):
        r"""Test creating output."""
        if ((not (fname_expected.endswith(('.png', '.gif'))
                  or disable_compare)
             and not os.path.isfile(fname_expected))):
            raise AssertionError(f'Expected \"{output}\" output '
                                 f'does not exist: {fname_expected}')
        instance.get_output(output)
        assert os.path.isfile(fname_actual)
        if disable_compare and not os.path.isfile(fname_expected):
            shutil.copyfile(fname_actual, fname_expected)
        if fname_expected.endswith(('.png', '.gif')) or disable_compare:
            # Don't compare png binaries
            return
        self.compare_output(output, fname_actual, fname_expected)

    def compare_output(self, output, actual, expected):
        r"""Compare output files.

        Args:
            output (str): Name of output contained in files.
            actual (str): Path to output produced by the test.
            expected (str): Path to output that is expected.

        """
        contents_actual = open(actual, 'rb').read()
        contents_expected = open(expected, 'rb').read()
        assert contents_actual == contents_expected


class TestLayoutTask(TestTask):
    r"""Class for testing layout."""

    _registry_name = 'layout'
    _params = {
        # 'crop': ['maize'],
        # 'id': ['B73_WT'],
        'arguments': {
            'values': [
                {'canopy': 'single'},
                {'canopy': 'single', 'periodic_canopy': True},
                {'canopy': 'unique', 'periodic_canopy': True},
                {'canopy': 'unique', 'time': 'noon'},
                {'canopy': 'tiled'},
            ],
            'ids': [
                'single',
                'single_periodic',
                'unique',
                'unique_periodic',
                'tiled',
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
        'arguments': {
            'values': [
                {'canopy': 'single'},
                {'crop': 'maize', 'id': 'B73_WT', 'canopy': 'unique'},
                {'crop': 'maize', 'id': 'B73_WT', 'canopy': 'tile'},
            ],
        },
    }

    @pytest.fixture(scope="class")
    def disable_compare(self, arguments):
        r"""bool: True if byte-wise comparison should be disabled."""
        if arguments.get('canopy', 'single') in ['unique', 'tile']:
            return True
        return self._disable_compare


class TestRayTraceTask(TestTask):
    r"""Class for testing raytrace."""

    _registry_name = 'raytrace'
    _params = {
        'arguments': {
            'values': [
                {'crop': 'maize', 'id': 'B73_WT',
                 'periodic_canopy': True},
            ],
        },
    }

    @pytest.fixture
    def disable_compare(self, output):
        r"""bool: True if byte-wise comparison should be disabled."""
        if output in ['raytrace_limits', 'traced_mesh']:
            return True
        return self._disable_compare

    @pytest.fixture
    def fname_actual(self, instance, output, fname_expected):
        r"""str: Path of file containing generated result."""
        if output == 'raytrace_limits':
            return fname_expected
        return instance.output_file(output)


class TestRender(TestTask):
    r"""Class for testing render."""

    _registry_name = 'render'

    @pytest.fixture
    def disable_compare(self, output):
        r"""bool: True if byte-wise comparison should be disabled."""
        if output == 'render_camera':
            return True
        return self._disable_compare

    @pytest.fixture
    def fname_actual(self, instance, output, fname_expected):
        r"""str: Path of file containing generated result."""
        if output == 'render_camera':
            return fname_expected
        return instance.output_file(output)


class TestTotals(TestTask):
    r"""Class for testing totals."""

    _registry_name = 'totals'
    _params = {
        'arguments': {
            'values': [
                {'crop': 'maize', 'id': 'B73_WT',
                 'periodic_canopy': True},
            ],
        },
    }


class TestAnimateTask(TestTask):
    r"""Class for testing animate."""

    _registry_name = 'animate'
    _params = {
        'arguments': {
            'values': [
                {'crop': 'maize', 'id': 'B73_WT',
                 'periodic_canopy': True},
            ],
        },
    }


class TestMatchQueryTask(TestTask):
    r"""Class for testing match query."""

    _registry_name = 'match_query'
    _params = {
        'crop': ['maize'],
        'id': ['rdla'],
    }


class TestPhotosynthesisTask(TestTask):
    r"""Class for testing photosynthesis calc."""

    _registry_name = 'photosynthesis'
