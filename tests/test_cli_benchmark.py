import argparse
from gemma_4_sql.cli import benchmark_cmd

def test_benchmark_cmd():
    args = argparse.Namespace(model="gemma-4", hardware="gpu", batch_size=1, backend="jax")
    benchmark_cmd(args)
