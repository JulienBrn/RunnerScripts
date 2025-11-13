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
    bin_size:  Annotated[str | None, Field(
        description="Size of the bins", 
        default="auto",
    )]
    bin_agg:  Annotated[str | None, Field(
        description="Method used to aggregate the bins",
        examples=["max", "mean", "median", "min"],
        default = "max"
    )]
    output_path: Annotated[Path, Field(
        description="Where you want to store the histogram results",
        examples=["/media/filer2/T4b/Temporary/....html"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".html"]))
    )]
    allow_output_overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="monkey", cpu=0.1, memory=2, gpu=0.0, disk=3)
    
args = Args()

from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import cv2

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    file = Path(args.timestamp_path)
    arr = np.loadtxt(file.open("r").readlines()[:-3]).astype(int)
    diff = arr[1:] - arr[:-1]
    df = pd.Series(diff, name="diff").rename_axis('frame_num').reset_index()
    # df = pd.DataFrame()
    # df["file"] = file

    # def get_fs(f):
    #     candidates = list(f.parent.glob("*.mp4"))
    #     if len(candidates) == 0:
    #         return -1
    #     elif len(candidates) == 1:
    #         return np.round(cv2.VideoCapture(candidates[0]).get(cv2.CAP_PROP_FPS), 2)
    #     else:
    #         raise Exception(f"Error, several videos for file {f}")
        
    # df["fps"] = df["file"].apply(get_fs)

    # all = []
    # for row in df.to_dict(orient="index").values():
    #     f = row["file"]
    #     arr = np.loadtxt(f.open("r").readlines()[:-3]).astype(int)
    #     diff = arr[1:] - arr[:-1]
    #     fdf = pd.Series(diff, name="diff").rename_axis('frame_num').reset_index()
    #     fdf = fdf.assign(**row)
    #     all.append(fdf)
    # all = pd.concat(all)

    fig = px.histogram(df, x="diff", nbins=100, labels={"diff": "time between consecutive frames (ms)"}, log_y=True)
    fig.write_html(output_path)


    # if fig_path.exists():
    #     print(f"Figure available at file://{fig_path}")


    # if a.bin_size == "auto":
    #     bin_size = int(len(all.index)/10**5)
    # else:
    #     bin_size = int(a.bin_size)
    # if bin_size < 1:
    #     bin_size = 1

    # all["group"] = (np.arange(len(all.index)) /bin_size).astype(int)

    # grouped = all.groupby(["group", "fps", "file"]).agg(a.bin_agg).reset_index()

    # fig = px.line(grouped, x="group", y="diff", facet_row="fps", color="file", labels={"diff": f"{a.bin_agg} time between consecutive frames (ms)"}, hover_data="frame_num")

    # fig_path = a.output_hist_path/f"evolution-bin_size={bin_size}-bin_agg={a.bin_agg}.html"
    # if fig_path.exists():
    #     if a.overwrite == "ask_user":
    #         r = "nasdfasd"
    #         while r not in ["yes", "no"]:
    #             r = input(f'Do you want to overwrite {fig_path} ?')
    #         if r=="yes":
    #             fig.write_html(fig_path)
    #     elif a.overwrite == "yes":
    #         fig.write_html(fig_path)
    # else:
    #     fig.write_html(fig_path)

    # if fig_path.exists():
    #     print(f"Figure available at file://{fig_path}")
