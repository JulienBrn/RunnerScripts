from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
        - Goal: Count the  distances between poly frames.
    """
    timestamp_path: Annotated[Path, Field(
        description="Path to the poly .txt file",
        examples=["/media/filer2/T4b/Temporary/....txt"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".txt"]))
    )]
    nbins:  Annotated[int | None, Field(
        description="Size of the bins", 
        default=None,
    )]
    output_path: Annotated[Path, Field(
        description="Where you want to store the histogram results",
        examples=["/media/filer2/T4b/Temporary/....html"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".html"]))
    )]
    allow_output_overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(cpu=0.1, memory=0.1, gpu=0.0, disk=0.6)
    
args = Args()

from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    file = Path(args.timestamp_path)
    arr = np.loadtxt(file.open("r").readlines()[:-3]).astype(int)
    diff = arr[1:] - arr[:-1]
    df = pd.Series(diff, name="diff").rename_axis('frame_num').reset_index()
    fig = px.histogram(df, x="diff", nbins=args.nbins, labels={"diff": "time between consecutive frames (ms)"}, log_y=True)
    fig.write_html(output_path)
