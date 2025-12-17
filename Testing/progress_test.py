from pathlib import Path
import subprocess
from pydantic import Field
from script2runner import CLI
from typing import Literal, List, ClassVar
from time import sleep
import tqdm

class Args(CLI):
    n: int = 1000
    t_sleep: float = 0.1
    use_tqdm: bool = False
    _run_info: ClassVar = dict(cpu=0.1, memory=0.1)

a = Args()

if a.use_tqdm:
    for i in tqdm.tqdm(range(a.n)):
        sleep(a.t_sleep)
else:
    for i in range(a.n):
        print(i)
        sleep(a.t_sleep)
    



