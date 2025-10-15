from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
import abc, json
import xarray as xr, numpy as np, pandas as pd


start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+ mk_or_pattern(start_path_patterns)+r'[^\\]*'+ mk_or_pattern(suffixes) + "$"



class Args(CLI):
    """
## Inputs
- **Excel files**:
  - `reference_event_path` → the reference event log
  - `relative_event_path` → the event log to align with the reference
- **Event mapping** → maps event names between files (e.g., `LED1` → `LED_1`)
- **Paths** for:
  - Output JSON file
  - Output HTML figures
  - Optional debug folder

## Workflow
1. **Load & preprocess events**
   - Reads Excel files
   - Aligns event names using the mapping
   - Ensures uniqueness and chronological ordering
   - Converts into an `xarray` dataset

2. **Initial synchronization**
   - Estimates approximate alignment parameters:
     - **slope** (time scaling)
     - **shift** (time offset)
     - **tolerance** (matching window)

3. **Event matching**
   - Matches events between reference and relative logs within tolerance

4. **Refined synchronization**
   - Applies least-squares fitting for optimal slope and shift

5. **Compute statistics**
   - Counts matched, missed, and partially matched events

6. **Generate visualization**
   - Creates interactive **Plotly** figures:
     - Matching overview
     - Time alignment trend
     - Event stats
   - Saves as HTML

7. **Save outputs**
   - **JSON** → contains final sync parameters (`shift`, `slope`)
   - **HTML** → interactive visualization report

## Outputs
- `output_path` → JSON file with sync results
- `figure_path` → HTML file with plots

---

👉 **In short:**  
This script synchronizes two sets of event logs, aligns their timelines, computes matching statistics, and produces both a JSON summary and an interactive HTML report.

    """
    reference_event_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....xlsx"], 
        description="Where you have your events", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xlsx"]))
    )]
    relative_event_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....xlsx"], 
        description="Where you have your events", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xlsx"]))
    )]
    event_mapping: Dict[str | int, str | int] = {"LED1": "LED_1", "LED2":"LED_2", "LED3":"LED_3"}
    tolerance: float = 0.020
    output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....json"], 
        description="Where you want your output json file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".json"]))
    )]
    figure_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....html"], 
        description="Where you want your output figures", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".html"]))
    )]
    debug_folder: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/...."], 
        description="Where you want your debug figures", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, []))
    )] | None = None
    
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)


a = Args()
a.event_mapping = {str(k):str(v) for k,v in a.event_mapping.items()}

import numpy as np, pandas as pd, xarray as xr
from typing import Literal, Annotated
import functools
import tqdm.auto as tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import xarray as xr, numpy as np, pandas as pd
import shutil
import plotly.colors as pc
from sync_helper import compute_initial_sync_values, compute_match_intervals
from sync_plotting_helper import plot_matching, plot_time_info, plot_type_info, get_plotly_config

def get_initial_event_ds():
    ref_df = pd.read_excel(a.reference_event_path)
    rel_df = pd.read_excel(a.relative_event_path)
    rel_df["event_name"] = rel_df["event_name"].astype(str).map({v: k for k, v in a.event_mapping.items()})

    for df in ref_df, rel_df:
        df["event_name"] = df["event_name"].astype(str)
        df.drop(df[~df["event_name"].isin(a.event_mapping.keys())].index, inplace = True)
        df["end"] = df["start"] + df["duration"]
        df["ev_index"] = df.assign(_count=1).groupby("event_name")["_count"].cumsum() -1
        if not "event_id" in df:
            df["event_id"] = df["event_name"] + "_#"+ df["ev_index"].astype(str)
        if df.duplicated("event_id").any():
            raise Exception("non unique event ids")
        if not (df["start"].shift(-1, fill_value=np.inf) >= df["start"]).all():
            raise Exception("Event dataframe is not sorted...") 
    all_ds = pd.concat([ref_df.assign(which="ref"), rel_df.assign(which="rel")]).to_xarray().rename(index="event")
    all_ds["t"] = all_ds[["start", "end"]].to_array(dim="bound")
    all_ds = all_ds.drop_vars(["start", "end"]).set_coords(["ev_index", "event_id", "event_name", "which"])[["t"]]
    all_ds = all_ds.sel(event = all_ds["t"].notnull().all("bound"))
    return all_ds

def compute_sync(all_ds):
    n, tol, slope, min_shift, max_shift = compute_initial_sync_values(
        [all_ds["t"].sel(bound=bound, event=(all_ds["event_name"] == ev) & (all_ds["which"] == "ref")).to_numpy() for bound in ("start", "end") for ev in a.event_mapping],
        [all_ds["t"].sel(bound=bound, event=(all_ds["event_name"] == ev) & (all_ds["which"] == "rel")).to_numpy() for bound in ("start", "end") for ev in a.event_mapping],
        tol_search=(0.001, a.tolerance, 10**-3, 0.98),
        slope_search=(0.9999,1.0001,10,10**-6)
    ) 
    return tol, slope, (min_shift+max_shift)/2
   
def compute_matching(all_ds, tol, slope, shift):
    def get_matching_for_event(ev):
        ref_arr_start = all_ds.sel(bound="start", event=(all_ds["event_name"] == ev) & (all_ds["which"] == "ref"))
        rel_arr_start = all_ds.sel(bound="start", event=(all_ds["event_name"] == ev) & (all_ds["which"] == "rel"))
        ref_arr_end = all_ds.sel(bound="end", event=(all_ds["event_name"] == ev) & (all_ds["which"] == "ref"))
        rel_arr_end = all_ds.sel(bound="end", event=(all_ds["event_name"] == ev) & (all_ds["which"] == "rel"))
        group_ds, ref_ev, rel_ev = compute_match_intervals(ref_arr_start["t"].to_numpy(), slope*rel_arr_start["t"].to_numpy()+shift,
                                ref_arr_end["t"].to_numpy(), slope*rel_arr_end["t"].to_numpy()+shift,
                                tol
        )
        ref_ev = xr.merge([ref_ev, all_ds.sel(event=(all_ds["event_name"] == ev) & (all_ds["which"] == "ref"))])
        rel_ev = xr.merge([rel_ev, all_ds.sel(event=(all_ds["event_name"] == ev) & (all_ds["which"] == "rel"))])
        evs=xr.concat([ref_ev, rel_ev], dim="event")
        group_ds = group_ds
        tmp_ref_ds = all_ds.sel(event=(all_ds["event_name"] == ev) & (all_ds["which"] == "ref")).isel(event=group_ds["ev_ind"].sel(which="ref")).assign(which="ref")
        tmp_rel_df = all_ds.sel(event=(all_ds["event_name"] == ev) & (all_ds["which"] == "rel")).isel(event=group_ds["ev_ind"].sel(which="rel")).assign(which="rel")
        all_tmp = xr.concat([tmp_ref_ds, tmp_rel_df], dim="which")
        all_tmp = xr.where(group_ds["ev_ind"] >= 0, all_tmp, np.nan)
        group_ds = xr.merge([group_ds, all_tmp])
        if not ((group_ds["ev_ind"] == group_ds["ev_index"]) | (group_ds["ev_ind"] == -1)).all():
            raise Exception("Problem")
        else:
            group_ds = group_ds.drop_vars("ev_index").rename(ev_ind="ev_index")
        return group_ds, evs
    group_ds = []
    event_ds = []
    match_group_inc = 0
    for ev in a.event_mapping:
        ds, evs = get_matching_for_event(ev)
        for d in ds, evs:
            d["match_group"] = d["match_group"] + match_group_inc
        match_group_inc+=ds.sizes["match_group"]
        group_ds.append(ds.assign(event_name=ev))
        event_ds.append(evs)
    group_ds = xr.concat(group_ds, dim="match_group")
    # group_ds = group_ds.assign(match_group=np.arange(group_ds.sizes["match_group"]))
    group_ds["has_bound"] = (group_ds["ev_index"]>=0).all("which")
    event_ds= xr.concat(event_ds, dim="event")
    return group_ds, event_ds


def sync_from_matching(matching_merged: xr.Dataset):
    data = matching_merged.stack(ev=["match_group", "bound"], create_index=False)
    data = data.sel(ev=data["t"].notnull().all("which"))
    rel_t = data["t"].sel(which="rel")
    ref_t = data["t"].sel(which="ref")
    lr, residuals, rank, sing = np.linalg.lstsq(np.stack([rel_t.to_numpy(), np.ones_like(rel_t)]).T, ref_t.to_numpy())
    slope = lr[0]
    shift=lr[1]
    tol = np.abs(ref_t-(rel_t*slope+shift)).max().item()
    return tol, slope, shift

def get_matching_stats(all_ds, group_ds: xr.Dataset):
    stats = xr.Dataset()
    if len(group_ds["event_name"].dims) ==0:
        group_ds = group_ds.assign_coords(event_name = group_ds["match_group"].astype(str).str.slice(0, 0) + group_ds["event_name"])
    stats["all_events"] = all_ds["t"].notnull().groupby(["which", "event_name"]).sum("event").astype(int)
    stats["primary_matches"] = (group_ds["has_bound"].all("bound")).fillna(0).groupby("event_name").sum("match_group")
    stats["events_in_holes"] = (group_ds["n_holes"]).fillna(0).groupby("event_name").sum("match_group")
    stats["no_matching_bound"] = group_ds["has_bound"].groupby("event_name").sum("match_group") - group_ds["has_bound"].all("bound").groupby("event_name").sum("match_group")
    stats["missed"] = (stats["all_events"]  - stats["no_matching_bound"] - stats["events_in_holes"] - stats["primary_matches"])
    try:
        stats = stats.fillna(0).astype(int)
    except:
        print("Could not convert stats to integer...")
        return stats
    return stats

def get_summary_figure_html(matching_name, events_matching_metadata, matching_info, stats, slope, shift, tol):
    if len(matching_info["event_name"].dims) ==0:
        matching_info = matching_info.assign_coords(event_name = matching_info["match_group"].astype(str).str.slice(0, 0) + matching_info["event_name"])
    stats = plot_type_info(stats)
    trend = plot_time_info(matching_info, slope, shift)
    matching = plot_matching(slope, shift, matching_info, events_matching_metadata, list(a.event_mapping.keys()))

    html=""
    html+='<div style="height:100vh;">'
    html+= f'<h3 style="text-align:center; width:100%;">{matching_name} matching from slope={slope}, shift={shift}, tol={tol}</h3>'
    html+='<div style="display:grid; gap:0;grid-template-columns: 100%; grid-template-rows: 30% 40% 30%;height:90%;">'

    l = [("stats", stats), ("trend", trend), ("matching", matching)]
    for j , (fname, fig) in enumerate(l):
        # if j < 2:
            html+=f'<div id="{fname}"></div>'
        # else:
        #     html+=f'<div id="{fname}" style="grid-column: span 2;"></div>'
    html+="</div></div>"
    for j, (fname, fig) in enumerate(l):
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=50),)
        html += fig.to_html(config=get_plotly_config(f'{matching_name}_{fname}_plot'), include_plotlyjs=j==0, div_id=fname)
    return html

def main():
    print("Loading initial dataframes")
    initial_event_ds = get_initial_event_ds()
    print(initial_event_ds)
    print(initial_event_ds["t"].groupby(["event_name", "which", "bound"]).count().rename("counts").to_dataframe().reset_index())
    print("Computing first sync parameters using approximation method")
    start_tol, start_slope, start_shift = compute_sync(initial_event_ds)
    print(start_tol, start_slope, start_shift)
    print("Computing matching from first sync approximation and auto determined tolerance")
    start_matching_grp, start_matching_ev = compute_matching(initial_event_ds, start_tol, start_slope, start_shift)
    print(start_matching_grp)
    print("Computing optimal sync from matching")
    match_tol, new_slope, new_shift= sync_from_matching(start_matching_grp)
    print(match_tol, new_slope, new_shift)
    print("Computing matching from optimal sync and user tolerance")
    new_matching_grp, new_matching_ev = compute_matching(initial_event_ds, a.tolerance, new_slope, new_shift)
    print(new_matching_grp)
    print("Computing stats from matching")
    stats = get_matching_stats(initial_event_ds, new_matching_grp)
    print(stats.to_dataframe())
    print("Creating summary figure")
    fig_html = get_summary_figure_html("final", new_matching_ev, new_matching_grp, stats, new_slope, new_shift, a.tolerance)
    data = dict(shift=new_shift, slope=new_slope)
    return data, fig_html

for output_path in [a.figure_path, a.output_path]:
    if output_path.exists():
        if a.overwrite =="yes":
            output_path.unlink()
        else:
            raise Exception(f"Output path {output_path} already exists")
    output_path.parent.mkdir(exist_ok=True, parents=True)

data, fig_html = main()
with a.output_path.open("w") as f:
    json.dump(data, f)
with a.figure_path.open("w") as f:
    f.write(fig_html)
        





