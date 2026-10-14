r"""Benchmark construction and access of plant parameter generators.

Measures the cost of constructing a :class:`PlantGenerator` from default
parameters, flattening its parameters (``all_parameters``), generating
its JSON ``schema`` and reading parameter values. This is used as the
performance baseline for the pydantic-based parameter refactor.

Example::

    python scripts/benchmark_parameters.py --crops maize
    python scripts/benchmark_parameters.py --crops maize tomato --repeat 2
"""
import time
import argparse

from canopy_factory import utils


def _timeit(fn, repeat=1):
    r"""Return the best wall-clock time (in seconds) to run a callable.

    Args:
        fn (callable): Callable taking no arguments.
        repeat (int, optional): Number of times to run the callable. The
            minimum time is returned.

    Returns:
        float: Best time in seconds.

    """
    best = None
    for _ in range(max(repeat, 1)):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


def benchmark_crop(cls, repeat=1, number=100):
    r"""Benchmark a single plant generator class.

    Args:
        cls (type): Plant generator class.
        repeat (int, optional): Number of times to repeat each timed
            operation. The minimum time is reported.
        number (int, optional): Number of accesses to perform when
            timing parameter reads.

    Returns:
        dict: Benchmark results for the crop.

    """
    out = {'crop': cls.__name__}
    out['construct'] = _timeit(cls, repeat=repeat)
    inst = cls()
    out['nparams'] = len(inst.all_parameters)

    def flatten():
        inst.__dict__.pop('all_parameters', None)
        return inst.all_parameters

    out['all_parameters'] = _timeit(flatten, repeat=repeat)
    out['schema'] = _timeit(lambda: cls.schema(), repeat=repeat)

    simple = list(inst.parameters.simple.keys())
    if not simple:
        simple = [k for k in inst.all_parameters if k != 'crop']
    keys = [simple[i % len(simple)] for i in range(number)]

    def access():
        for k in keys:
            inst.get(k, None)

    out['n_access'] = len(keys)
    out['access'] = _timeit(access, repeat=repeat)
    return out


def main(crops=None, repeat=1, number=100):
    r"""Benchmark the requested crops and print a summary table.

    Args:
        crops (list, optional): Names of crops to benchmark. Defaults to
            ``['maize']``.
        repeat (int, optional): Number of times to repeat each timed
            operation.
        number (int, optional): Number of parameter reads to time.

    """
    if crops is None:
        crops = ['maize']
    results = []
    for name in crops:
        cls = utils.get_class_registry().get('crop', name)
        print(f'Benchmarking "{name}" ...', flush=True)
        results.append(benchmark_crop(cls, repeat=repeat, number=number))
    header = (
        f'{"crop":<22} {"nparams":>8} {"construct(s)":>13} '
        f'{"flatten(s)":>11} {"schema(s)":>11} {"access(us/op)":>14}'
    )
    print()
    print(header)
    print('-' * len(header))
    for res in results:
        us_per_op = (
            1e6 * res['access'] / res['n_access']
            if res['n_access'] else float('nan')
        )
        print(
            f'{res["crop"]:<22} {res["nparams"]:>8} '
            f'{res["construct"]:>13.4f} {res["all_parameters"]:>11.4f} '
            f'{res["schema"]:>11.4f} {us_per_op:>14.3f}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--crops', type=str, nargs='+', default=['maize'],
        help='Name(s) of the crops to benchmark (default: maize)',
    )
    parser.add_argument(
        '--repeat', type=int, default=1,
        help='Number of times to repeat each timed operation (default: 1)',
    )
    parser.add_argument(
        '--number', type=int, default=100,
        help='Number of parameter reads to time per access pass '
             '(default: 100)',
    )
    args = parser.parse_args()
    main(crops=args.crops, repeat=args.repeat, number=args.number)
