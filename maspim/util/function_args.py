import inspect
from typing import Callable


def get_arg_names_of_func(func: Callable, only_kwargs: bool = False) -> list[str]:
    """obtain the kwarg and/or arg names of a function"""
    params = inspect.signature(func).parameters
    arg_names = list(params.keys())
    if not only_kwargs:
        return arg_names
    arg_vals = [p.default for p in params.values()]
    kwarg_names = [a for a, v in zip(arg_names, arg_vals) if v != inspect._empty]
    return kwarg_names
