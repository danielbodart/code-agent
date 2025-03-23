#!/usr/bin/env ./commands.sh
import sh # type: ignore
import os
from datetime import UTC, datetime
import sys
import unittest

def version():
    branch = os.getenv('CIRCLE_BRANCH', sh.git('rev-parse', '--abbrev-ref', 'HEAD').strip())
    build_number = os.getenv('CIRCLE_BUILD_NUM', datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S'))
    revisions = sh.git('rev-list', '--count', branch).strip()
    version = f"0.{revisions}.{build_number}"
    print(f"version: {version}")


def clean():
    print(sh.rm('-rf', 'artifacts', _err_to_out=True))


def check():
    sh.mypy('src', '--ignore-missing-imports', _err_to_out=True)


def test(pattern = 'test_*.py'):
    loader = unittest.TestLoader()
    suite = loader.discover('test', pattern)
    runner = unittest.TextTestRunner()
    runner.run(suite)


def build():
    clean()
    check()
    test()


def train(*args):
    check()
    print(sh.uv('run', 'src/train_model.py', *args, _err_to_out=True))


def train_diffusion(*args):
    check()
    print(sh.uv('run', 'src/train_diffusion.py', *args, _err_to_out=True))


def predict(*args):
    check()
    print(sh.uv('run', 'src/predict_model.py', *args, _err_to_out=True))


def predict_diffusion(*args):
    check()
    print(sh.uv('run', 'src/predict_diffusion.py', *args, _err_to_out=True))


def tune(*args):
    check()
    print(sh.uv('run', 'src/hyperparamter-tune.py', *args, _err_to_out=True))


def simple(*args):
    check()
    sh.uv('run', 'src/simple_masked_model.py', *args, _out=sys.stdout, _err=sys.stderr)


def modern_bert(*args):
    check()
    sh.uv('run', 'src/modern-bert-test.py', *args, _out=sys.stdout, _err=sys.stderr)


def bert(*args):
    check()
    sh.uv('run', 'src/predict_masked_diffusion_bert.py', *args, _out=sys.stdout, _err=sys.stderr)

def train_bert(*args):
    check()
    sh.uv('run', 'src/train_masked_diffusion_bert.py', *args, _out=sys.stdout, _err=sys.stderr)

def ci():
    build()


if __name__ == "__main__":
    func_name = 'build' if len(sys.argv) == 1 else sys.argv[1]
    args = sys.argv[2:]


    if func_name in globals():
        func = globals()[func_name]
        if callable(func):
            try:
                func(*args)
            except sh.ErrorReturnCode as e:
                print(e.stdout.decode())
                sys.exit(1)
        else:
            print(f"Invalid target '{func_name}'")
    else:
        try:
            print(sh.uv('run', *sys.argv[1:], _err_to_out=True))
        except sh.ErrorReturnCode as e:
            print(e.stdout.decode())
            sys.exit(e.exit_code)
