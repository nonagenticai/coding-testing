"""A tiny real library for library behaviour parity through lib_call_harness.py."""


def add(a, b):
    return a + b


def divide(a, b):
    return a / b


def tags(*names, sep=","):
    return {"joined": sep.join(names), "count": len(names)}


def noisy(x):
    print("this goes to stderr, not into the verdict")
    return [x, x]


def as_set(*items):
    return set(items)


class Calc:
    @staticmethod
    def double(x):
        return 2 * x
