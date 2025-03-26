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
    sh.rm('-rf', 'artifacts', _out=sys.stdout, _err=sys.stderr)


def check():
    sh.mypy('src', '--ignore-missing-imports', _out=sys.stdout, _err=sys.stderr)


def test(test_name = None):
    if test_name is None:
        sh.uv('run', 'python', '-m', 'unittest', 'discover', 'test', _out=sys.stdout, _err=sys.stderr)
    else:
        sh.uv('run', 'python', '-m', 'unittest', test_name, _out=sys.stdout, _err=sys.stderr)


def build():
    clean()
    check()
    test()


def predict(*args):
    check()
    sh.uv('run', 'src/predict.py', *args, _out=sys.stdout, _err=sys.stderr)


def train(*args):
    check()
    sh.uv('run', 'src/train.py', *args, _out=sys.stdout, _err=sys.stderr)


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
            sh.uv('run', *sys.argv[1:], _out=sys.stdout, _err=sys.stderr)
        except sh.ErrorReturnCode as e:
            print(e.stdout.decode())
            sys.exit(e.exit_code)
