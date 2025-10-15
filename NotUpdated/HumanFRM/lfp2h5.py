from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re, shutil


start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+ mk_or_pattern(start_path_patterns)+r'[^\\]*'+ mk_or_pattern(suffixes) + "$"

class Args(CLI):
    """
        # LFP JSON to Zarr Conversion Script — Summary

        This code reads a **Medtronic PERCEPT LFP JSON file**, extracts the **time-domain signals** from each channel, groups them by recording date, and saves each group as a **Zarr dataset** for analysis.

        ## Key Steps

        1. **Output Handling**  
        - Deletes the output folder if `overwrite="yes"`.

        2. **Signal Conversion**  
        - Converts each channel’s signal into an `xarray.DataArray` with a proper time axis and sampling frequency.

        3. **Grouping by Segment**  
        - Groups all channels from the same recording segment/date into one dataset.

        4. **Saving Output**  
        - Saves each segment as a separate `.xr.zarr` file using a formatted file name.

        ## Summary

        **Purpose:** Convert raw LFP JSON recordings into structured, time-indexed Zarr files suitable for analysis.

    """
    lfp_path: Annotated[Path, Field(
        description="Path to the file contining the lfp recording",
        examples=["/media/t4user/data1/Data/SpikeSorting/....json"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".json"]))
    )]
    output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....myfolder"], 
        description="Where you want your output files", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [""]))
    )]
    file_name_fstring: str = "{segnum}--{segdate}.xr.zarr"
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)
    
args = Args()

if args.output_path.exists():
    if args.overwrite =="yes":
        shutil.rmtree(args.output_path)
    else:
        raise Exception("Output path already exists")
    
import xarray as xr, pandas as pd, numpy as np, json, datetime
from dateutil import parser

args.output_path.parent.mkdir(exist_ok=True, parents=True)
lfp = json.load(args.lfp_path.open("r"))

date = parser.parse(lfp["SessionDate"]).astimezone(tz=None)

sigs = []
for d in lfp["IndefiniteStreaming"] if "IndefiniteStreaming" in lfp else lfp["BrainSenseTimeDomain"]:
    channel = d["Channel"]
    a = xr.DataArray(d["TimeDomainData"], dims="t")
    fs = d["SampleRateInHz"]
    a["t"] = np.arange(a.sizes["t"])/fs
    a["t"].attrs["fs"] = fs
    adate = str(date + datetime.timedelta(seconds=d["FirstPacketDateTimeOffsetInSeconds"]))
    sigs.append(dict(ar=a, channel=channel, date=adate))

def make_ds(g: pd.DataFrame) -> xr.Dataset:
    ds = xr.Dataset()
    for _, row in g.iterrows():
        ds[row["channel"]] = row["ar"]
    
    ret =  ds.to_array(dim="channel").to_dataset(name="data")
    ret.attrs["date"] = g["date"].iat[0]
    return ret

all_ds = []
for i, (_, g) in enumerate(pd.DataFrame(sigs).groupby("date", sort=True)):
    ds = make_ds(g).assign_attrs(segnum=i)
    all_ds.append(ds)

for ds in all_ds:
    print(ds)
    ds.to_zarr(args.output_path/args.file_name_fstring.format_map(ds.attrs))