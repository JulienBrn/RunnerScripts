from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import Literal, List


class Args(CLI):
    my_param_int: int
    my_param_bool: bool
    my_param_str: str
    my_intopt_param: int = 5

a: BaseModel = Args()

print(a.my_param_int, a.my_param_bool, a.my_param_str, a.my_intopt_param)
    



