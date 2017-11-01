from __future__ import print_function
import os, sys
from contextlib import contextmanager
@contextmanager
def RedirectStdout(newStdout):
    savedStdout, sys.stdout = sys.stdout, newStdout
    try:
        yield
    finally:
        sys.stdout = savedStdout

if __name__ == "__main__":
    with open("test.txt","a+") as f:
        print("haha")
        with RedirectStdout(f):
            print("a")
            print("b")
        print("c")
