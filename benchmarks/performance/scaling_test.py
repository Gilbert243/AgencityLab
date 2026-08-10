"""Compatibility entry point for the maintained v0.8 performance benchmark."""

try:
    from .benchmark_v08 import main, run_benchmark
except ImportError:  # Direct ``python benchmarks/performance/scaling_test.py`` use.
    from benchmark_v08 import main, run_benchmark


def run():
    """Run the CI-sized benchmark and return its structured report."""
    return run_benchmark(quick=True)


if __name__ == "__main__":
    raise SystemExit(main())
