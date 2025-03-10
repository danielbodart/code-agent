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
    print(sh.mypy('src', _err_to_out=True))


def test(pattern = 'test_*.py'):
    loader = unittest.TestLoader()
    suite = loader.discover('test', pattern)
    runner = unittest.TextTestRunner()
    runner.run(suite)


def build():
    clean()
    check()
    test()


def start(*args):
    print(sh.uv('run', 'src/main.py', *args, _err_to_out=True))


def ci():
    build()

# Add any additional logic or functions as needed.

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
