from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
import abc
import xarray as xr, numpy as np, pandas as pd
import dask.diagnostics

start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+ mk_or_pattern(start_path_patterns)+r'[^\\]*'+ mk_or_pattern(suffixes) + "$"


class Loader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> xr.DataArray:...

class XarrayLoader(Loader, BaseModel):
    input_method: Literal["xarray"] = "xarray"
    input_path: Annotated[Path, Field(
        description="Path to the file contining the data from which to extract events",
        examples=["/media/t4user/data1/Data/SpikeSorting/....xr.zarr"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.h5", ".xr.zarr"]))
    )]
    data_array_name: str | None = None
    load_method: Literal["h5", "zarr", "auto"] = "auto"
    slice_start : float | None = None
    slice_end : float | None = None

    def load(self) -> xr.DataArray:
        if self.load_method == "auto":
            if self.input_path.suffix == ".zarr":
                load_method="zarr"
            elif self.input_path.suffix == ".h5":
                load_method="h5"
            else:
                raise Exception("Unknown load method")
        else:
            load_method = self.load_method

        if load_method == "zarr":
            ds= xr.open_zarr(self.input_path)
        elif load_method == "h5":
            ds= xr.open_dataset(self.input_path)
        else:
            raise Exception("Unknown load method")
        
        display(ds)
        with dask.diagnostics.ProgressBar():
            ds = ds.sel(t=slice(self.slice_start, self.slice_end)).compute()
        display(ds)
        if self.data_array_name is None:
            if "__xarray_dataarray_variable__" in ds:
                return ds["__xarray_dataarray_variable__"]
            else:
                raise Exception("Array name needs to be provided")
        else:
            return ds[self.data_array_name]

class Thresholder(abc.ABC, BaseModel):
    @abc.abstractmethod
    def get_thresholds(self, arr) -> xr.DataArray:...
        

class AutoThresholdingMethod(Thresholder, BaseModel):
    threshold_method: Literal["auto"] = "auto"
    per_channel: bool = True
    dist_output_path: None | Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....html"], 
        description="Where you want your output distribution information", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".html"]))
    )] = None
    def get_thresholds(self, arr) -> xr.DataArray:
        import numpy as np
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        if self.dist_output_path is not None:
            self.dist_output_path.parent.mkdir(exist_ok=True, parents=True)
            figure_info = []

        def get_threshold(arr: np.ndarray, channel):
            positive_peaks, _ = find_peaks(arr)
            negative_peaks, _ = find_peaks(-arr)
            arr = arr[np.concatenate([positive_peaks, negative_peaks])]
            quantiles = np.quantile(arr, [0.0001, 0.9999])
            arr = arr[(arr >=quantiles[0]) & (arr <=quantiles[1])]
            if arr.size > 10**5:
                arr = np.random.default_rng().choice(arr, 10**5)
            min = np.min(arr)
            max = np.max(arr)
            eval = np.linspace(min, max, 1000, endpoint=True)
            bw_value = arr.size**(-1./5)
            min_bw = 0
            max_bw = None
            n_tries = 0
            last_good_thresh = None
            while n_tries < 8:
                n_tries+=1
                dist = gaussian_kde(arr, bw_method=bw_value)(eval)
                
                peaks, _ = find_peaks(dist)
                peaks = list(peaks)
                if dist[0] > dist[1]:
                    peaks = [0] + peaks
                if dist[-1] > dist[-2]:
                    peaks =peaks + [-1]
                peaks = np.array(peaks)
                figure_info.append(dict(channel=channel, iteration=n_tries, bw=bw_value, x_times=eval, dist=dist, peaks_indices=peaks))
                print(f"[Channel {channel}, Test #{n_tries}] Testing bandwith kde of {bw_value}. Got {peaks.size} peaks")
                if peaks.size > 2:
                    min_bw = bw_value
                    if max_bw is None:
                        bw_value*=2
                    else:
                        bw_value= (bw_value+max_bw)/2
                elif peaks.size == 2:
                    pos = dist[peaks[0]:peaks[1]].argmin()
                    last_good_thresh = eval[pos + peaks[0]]
                    max_bw = bw_value
                    bw_value= (bw_value+min_bw)/2
                else:
                    max_bw = bw_value
                    bw_value= (bw_value+min_bw)/2
            return last_good_thresh

        if self.per_channel:
            ret = xr.apply_ufunc(get_threshold, arr, arr["channel"], input_core_dims=[["t"], []], output_core_dims=[[]], vectorize=True)
        else:
            threshold = get_threshold(arr.to_numpy().flatten(), "all")
            ret = xr.ones_like(arr["channel"]) * threshold
        if self.dist_output_path is not None:
            dfs = []
            for fi in figure_info:
                df = pd.DataFrame()
                df["amp"] = fi["x_times"]
                df["kde"] = fi["dist"]
                df["bw"] = fi["bw"]
                df["iteration"] = fi["iteration"]
                df["channel"] = fi["channel"]
                df["n_peaks"] = len(fi["peaks_indices"])
                dfs.append(df)
            dist = pd.concat(dfs)
            fig = px.line(dist, x="amp", y="kde", facet_row="channel", color="bw", line_dash="n_peaks", hover_data=["bw", "n_peaks", "iteration"])
            fig.write_html(self.dist_output_path)
        return ret


class ManualThresholdingMethod(Thresholder, BaseModel):
    threshold_method: Literal["manual"] = "manual"
    threshold_default_value: float
    threshold_channel_value: Dict[str, float] = {}
    def get_thresholds(self, arr) -> xr.DataArray:
        def get_threshold(chan):
            return self.threshold_channel_value.get(chan, self.threshold_default_value)
        return xr.apply_ufunc(get_threshold, arr["channel"], vectorize=True)

class PreprocessMethod(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, arr) -> xr.DataArray:...

class NormalizePreprocessMethod(PreprocessMethod, BaseModel):
    preprocessing_method: Literal["normalize"] = "normalize"
    normalize_channel: str
    def preprocess(self, arr) -> xr.DataArray:
        norm = arr.sel(channel=self.normalize_channel)
        return (arr.drop_sel(channel=self.normalize_channel)-norm)/norm
    
class ArtefactsPreprocessMethod(PreprocessMethod, BaseModel):
    preprocessing_method: Literal["hartefacts"] = "hartefacts"
    def preprocess(self, arr: xr.DataArray) -> xr.DataArray:
        arr = np.abs(arr)
        arr = (arr - arr.mean("t"))/arr.std("t")
        arr = arr.sum("channel")
        arr = arr.expand_dims("channel")
        arr["channel"] = xr.DataArray(["summed_channel"], dims="channel")
        return arr
    

class FilterPreprocessMethod(PreprocessMethod, BaseModel):
    preprocessing_method: Literal["filter"] = "filter"
    low_freq: float | None = None
    high_freq: float | None = None
    filter_type: Literal["lowpass", "highpass", "bandpass"]

    def preprocess(self, arr: xr.DataArray) -> xr.DataArray:
        import scipy.signal
        fs = 1/arr["t"].diff("t").mean().item()
        if self.filter_type == "lowpass":
            filter = scipy.signal.butter(3, self.low_freq, btype="lowpass", output="sos", fs=fs)
        elif self.filter_type == "highpass":
            filter = scipy.signal.butter(3, self.high_freq, btype="highpass", output="sos", fs=fs)
        elif self.filter_type == "bandpass":
            filter = scipy.signal.butter(3, [self.low_freq, self.high_freq], btype="bandpass", output="sos", fs=fs)
        arr = xr.apply_ufunc(lambda a: scipy.signal.sosfiltfilt(filter, a, axis=-1), arr, input_core_dims=[["t"]], output_core_dims=[["t"]])
        return arr
    
class PolyTimestampsPreprocessMethod(PreprocessMethod, BaseModel):
    preprocessing_method: Literal["poly_timestamps"] = "poly_timestamps"
    timestamps_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....txt"], 
        description="Where the video timestamps generated by poly are", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".txt"]))
    )]
    def preprocess(self, arr) -> xr.DataArray:
        import numpy as np
        time_stamps = np.genfromtxt(self.timestamps_path, skip_footer=3)
        res = arr.copy()
        if np.abs(len(time_stamps) - res.sizes["t"]) > 5:
            raise Exception(f"Sizes of timestamps in correction ({len(time_stamps)}) do not match number of timestamps in data ({res.sizes['t']})")
        elif len(time_stamps) > res.sizes["t"]:
                print(f"Removing some timestamps in correction")
                time_stamps=time_stamps[:res.sizes["t"]]
        elif len(time_stamps) < res.sizes["t"]:
            print(f"Removing some timestamps in data")
            res = res.isel(t=slice(0, len(time_stamps)))
        res["t"] = time_stamps/1000
        return res
    
class Args(CLI):
    """
        # EEG/LFP Event Detection Pipeline

This script implements a **data processing pipeline for EEG/LFP event detection**.

---

## 🗂 File Loading
- Loads time-series data (EEG/LFP) from:
  - `.zarr`
  - `.h5` files  
- Uses **xarray** for dataset handling.

---

## ⚙️ Preprocessing Options
Supports several preprocessing methods:
- **Normalization**: Relative to a reference channel  
- **Artifact detection**: Z-scoring and channel summation  
- **High-pass filtering**: Removes low-frequency drift  
- **Timestamp alignment**: Sync with external timestamps (e.g., Poly video)  

---

## 📉 Thresholding Methods
Two approaches:
- **Automatic**:
  - Kernel Density Estimation (KDE) + peak detection
  - Determines thresholds per channel
- **Manual**:
  - User-specified default or per-channel thresholds

---

## 🔎 Event Detection
- Compares signal against thresholds → binary above/below threshold
- Detects **rising** and **falling** edges (event start & end)
- Optional filtering of events too close together (`min_distance`)
- Creates a structured event table:
  - `event_name` (channel)
  - `start` (time)
  - `duration`

---

## 💾 Outputs
1. **Excel file** (`.xlsx`)  
   - Contains the event table (channel, start, duration)  
2. **Interactive Plotly HTML** (`.html`)  
   - Shows:
     - Raw signals per channel
     - Threshold lines
     - Highlighted event windows

---

👉 **In short:**  
The script **loads EEG/LFP data, preprocesses it, applies thresholding to detect events, saves results to Excel, and generates interactive plots for inspection.**

    """
    input: Annotated[XarrayLoader, Field(
        description="How to load the data",
    )]
    preprocessing: Annotated[List[NormalizePreprocessMethod | PolyTimestampsPreprocessMethod | ArtefactsPreprocessMethod | FilterPreprocessMethod], Field(default=[], description="Preprocessing to apply (t_modification, normalization (df/f), ...)")]
    threshold: Annotated[AutoThresholdingMethod | ManualThresholdingMethod, Field(
        description="How to threshold the continuous data"
    )]
    min_distance: float | None = None
    output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....xlsx"], 
        description="Where you want your output excel file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xlsx"]))
    )]
    fig_output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....html"], 
        description="Where you want your output figure", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".html"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)

a = Args()

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shutil

for output_path in [a.output_path, a.fig_output_path]:
    if output_path is None:
        if output_path.exists():
            if a.overwrite =="yes":
                shutil.rmtree(output_path)
            else:
                raise Exception(f"Output path {output_path} already exists")
        output_path.parent.mkdir(exist_ok=True, parents=True)


display = print
print("Loading data, this may take a while")
ds = xr.Dataset()
input = a.input.load()
for preprocessing in a.preprocessing:
    display(input)
    input = preprocessing.preprocess(input)
display(input)

ds["input"] = input
ds["thresh"] = a.threshold.get_thresholds(ds["input"])
ds["binary"] = ds["input"] > ds["thresh"]
ds["is_rising"] = ds["binary"] > ds["binary"].shift(t=1, fill_value=True) 
ds["is_falling"] = ds["binary"].shift(t=-1, fill_value=True) < ds["binary"]


result = []
for chan in range(ds["channel"].size):
    d = ds.isel(channel=chan)

   
    def get_time_intersect_thresh(indice_above, indices_under):
        x1 = d["t"].isel(t=indices_under).to_numpy()
        y1 = d["input"].isel(t=indices_under).to_numpy()
        x2 = d["t"].isel(t=indice_above).to_numpy()
        y2 = d["input"].isel(t=indice_above).to_numpy()
        prop =(d["thresh"].item() - y1)/(y2 - y1)
        final_x = x1 + (x2 - x1)*prop
        return final_x
    
    rise_indices = np.flatnonzero(d["is_rising"].to_numpy())
    rise_times = get_time_intersect_thresh(rise_indices, rise_indices-1)
    fall_indices = np.flatnonzero(d["is_falling"].to_numpy())
    fall_times = get_time_intersect_thresh(fall_indices, fall_indices+1)

    if d["binary"].isel(t=0):
        rise_times = np.insert(rise_times, 0, np.nan)

    if d["binary"].isel(t=-1):
        fall_times = np.append(fall_times, np.nan)
    
    if rise_times.size != fall_times.size:
        raise Exception("Not same size...")
    if a.min_distance:
        new_rise_times = [rise_times[0]]
        new_fall_times = []

        for i in range(len(rise_times)-1):
            if (rise_times[i+1] - fall_times[i]) > a.min_distance:
                new_fall_times.append(fall_times[i])
                new_rise_times.append(rise_times[i+1])

        new_fall_times.append(fall_times[-1])
        rise_times = np.array(new_rise_times)
        fall_times = np.array(new_fall_times)

    duration = fall_times - rise_times
    df = pd.DataFrame().assign(start=rise_times, duration=duration, event_name=d["channel"].item())
    result.append(df)
result = pd.concat(result, ignore_index=True)[["event_name", "start", "duration"]]

print(result.groupby("event_name").size())

if ds.sizes["t"] > 10**5:
    plot_ds = ds.coarsen(t=int(ds.sizes["t"] / (10**5)), boundary="trim").max()
else:
  plot_ds = ds

result.sort_values("start").to_excel(a.output_path, index=False)

fig = make_subplots(rows=plot_ds["channel"].size, cols=1, shared_xaxes=True)
for chan in range(plot_ds["channel"].size):
  chan_ds = plot_ds.isel(channel=chan)
  m, M = chan_ds["input"].min().item(), chan_ds["input"].max().item()
  fig.add_trace(go.Scatter(x=chan_ds["t"], y=chan_ds["input"], name=chan_ds["channel"].item(), opacity=0.5), row=chan+1, col=1)
  fig.add_hline(y=chan_ds["thresh"].item(), line_dash="dot",
              annotation_text=str(chan_ds["channel"].item()), 
              annotation_position="bottom right", row=chan+1, col=1)
  sub_df: pd.DataFrame = result.loc[result["event_name"] == chan_ds["channel"].item()]
  for _, row in sub_df.iterrows():
    fig.add_trace(
      go.Scatter(
          x=[row["start"], row["start"] + row["duration"], row["start"] + row["duration"], row["start"]], 
          y=[m, m, M, M],
          fill="toself",
          fillcolor="pink",
          opacity=0.5,
          mode="lines",
          line=dict(width=0), showlegend=False), row=chan+1, col=1)
fig.write_html(a.fig_output_path)
