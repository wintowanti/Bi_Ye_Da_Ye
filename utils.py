from __future__ import print_function
import os, sys
from contextlib import contextmanager
from time import strftime, localtime, sleep

@contextmanager
def RedirectStdout(newStdout):
    savedStdout, sys.stdout = sys.stdout, newStdout
    try:
        yield
    finally:
        sys.stdout = savedStdout

def get_nowtime_str():
    nowtime_str = strftime("%Y-%m-%d %H:%M:%S", localtime())
    return nowtime_str

def test_redirectstdout():
    with open("test.txt","a+") as f:
        print("haha")
        with RedirectStdout(f):
            print("a")
            print("b")
        print("c")

def test_now_time():
    print(get_nowtime_str())

def judge():
    sleep(5)

if __name__ == "__main__":
    test_now_time()
