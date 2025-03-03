#!/usr/bin/env ./commands.sh
import sh # type: ignore
import os
from datetime import UTC, datetime
import sys

def version():
    branch = os.getenv('CIRCLE_BRANCH', sh.git('rev-parse', '--abbrev-ref', 'HEAD').strip())
    build_number = os.getenv('CIRCLE_BUILD_NUM', datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S'))
    revisions = sh.git('rev-list', '--count', branch).strip()
    version = f"0.{revisions}.{build_number}"
    print(f"version: {version}")


def clean():
    sh.rm('-rf', 'artifacts', _err_to_out=True)


def check():
    sh.mypy('', _err_to_out=True)


def test(*args):
    sh.python('-m', 'unittest', 'discover', '-s', 'test', *args, _err_to_out=True)


def build():
    clean()
    check()
    test()


def start(*args):
    print('TODO start', *args)


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
            sh.uv('run', *sys.argv[1:], _err_to_out=True)
        except sh.ErrorReturnCode as e:
            print(e.stdout.decode())
            sys.exit(e.exit_code)
