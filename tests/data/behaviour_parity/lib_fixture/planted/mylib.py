"""mylib.py with planted drift: add is off by one, divide raises ValueError."""


def add(a, b):
    return a + b + 1


def divide(a, b):
    if b == 0:
        raise ValueError("b must not be zero")
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
