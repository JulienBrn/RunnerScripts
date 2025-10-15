import json, shutil
from pathlib import Path
import tqdm.auto as tqdm
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from pydantic import Field, BaseModel
from script2runner import CLI

start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+mk_or_pattern(start_path_patterns)+r'[^\\]*'+mk_or_pattern(suffixes) + "$"


class Args(CLI):
    """
        - Goal: Get average luminosity of shapes in a video. The shapes are defined by the labelstudio annotations 
        - Output: an xarray of num_frames*num_shapes
    """
    video_path: Annotated[Path, Field(
        description="Path to the video", 
        examples=["/media/filer2/T4b/....mp4"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".mp4"]))
    )]
    output_path: Annotated[Path | None,  Field(
        description="Path to the result, may be empty if only image is demanded",
        examples=["/media/filer2/T4b/...xr.zarr"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.zarr"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    base_annotation_folder: Annotated[Path, Field(
        description="Path to the annotation project of label studio containing the annotations. Should probably not change",
        default = Path(r"/media/filer2/T4b/Labeling/Projet_LEDs/Annotations"))]
    annotation_num: Annotated[int,  Field(
        description="Annoation number in label studio",
        examples=[1797], 
    )]
    max_n_frames: Annotated[int | None,  Field(
        default=None,
        description="max number of frames to process, usually used for testing",
    )]
    fig_output_path: Annotated[Path | None,  Field(
        description="Path to the figure containing the first image with mask, may be None",
        examples=["/media/filer2/T4b/....html"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".html"]))
    )]
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)

a = Args()

import cv2
import plotly.express as px
import numpy as np, pandas as pd, xarray as xr
# base_folder = Path(r"/media/filer2/T4b/Datasets/Monkey/TestDataset")
# base_annotation_folder = Path(r"/media/filer2/T4b/Labeling/Projet_LEDs/Annotations")
# annotation_num = 1797
# video_path = base_folder / "Data/CTL/subject_Den/session_250716/videos/view_right/250716_Den_01.mp4"
# fig_output_path = None
# output_path = Path("mytest.xr.zarr")
# max_n_frames = 10**4

with (a.base_annotation_folder / f"{a.annotation_num}").open("r") as f:
    data = json.load(f)
led_info = {}
for item in data["result"]:
    label = item["value"]["ellipselabels"][0]
    led_info[label] = {"x_per": item["value"]["x"], "y_per": item["value"]["y"], "radiusX_per": item["value"]["radiusX"], "radiusY_per": item["value"]["radiusY"]}
leds = pd.DataFrame(led_info).T.to_xarray().rename(index="led_name")
leds

cap = cv2.VideoCapture(a.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
image = xr.Dataset()
image["y"] = xr.DataArray(np.arange(h), dims="y")
image["x"] = xr.DataArray(np.arange(w), dims="x")
image["mask"] = ((image["x"] - leds["x_per"]*w/100)**2/(leds["radiusX_per"]*w/100)**2 + (image["y"] - leds["y_per"]*h/100)**2/(leds["radiusY_per"]*h/100)**2) < 1
image

if a.fig_output_path:
    cap = cv2.VideoCapture(a.video_path)
    ret, frame = cap.read()
    cap.release()
    fig = px.imshow(frame)
    image["color"] = xr.DataArray(["r", "g", "b", "a"], dims="color")
    image["mask_color"] = xr.DataArray([0, 200, 0, 0.5], dims="color")
    rgba_mask = image["mask"] * image["mask_color"]
    import plotly.graph_objects as go
    for i in range(rgba_mask.sizes["led_name"]):
        fig.add_trace(go.Image(z=rgba_mask.isel(led_name=i).transpose("y", "x", "color"), colormodel="rgba"))
    fig.write_html(a.fig_output_path)

if a.output_path is None:
    exit(0)
if a.output_path.exists():
    if not a.overwrite:
        raise Exception("Path already exists")
    else:
        shutil.rmtree(a.output_path)
#Highly optimized code part, we convert everything to basic numpy and list, taking care of ordering
n_leds = image.sizes["led_name"]
mask_low_x = image["x"].where(image["mask"].any("y")).min("x").astype(int).to_numpy().tolist()
mask_high_x = (image["x"].where(image["mask"].any("y")).max("x").astype(int).to_numpy()+1).tolist()
mask_low_y = image["y"].where(image["mask"].any("x")).min("y").astype(int).to_numpy().tolist()
mask_high_y = (image["y"].where(image["mask"].any("x")).max("y").astype(int).to_numpy()+1).tolist()
cropped_masks = [image["mask"].isel(led_name=i).transpose("y", "x").to_numpy()[mask_low_y[i]:mask_high_y[i], mask_low_x[i]:mask_high_x[i]] for i in range(n_leds)]
mask_low_x, mask_high_x, mask_low_y, mask_high_y

cap = cv2.VideoCapture(a.video_path)
luminosities = []

if a.max_n_frames is None: 
    max_n_frames = num_frames
else:
    max_n_frames = min(a.max_n_frames, num_frames)

for i in tqdm.tqdm(range(max_n_frames), desc="Reading frames"):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lum = [np.sum(np.where(cropped_masks[i], gray[mask_low_y[i]:mask_high_y[i], mask_low_x[i]:mask_high_x[i]], 0)) for i in range(n_leds)]
    luminosities.append(lum)

cap.release()
#End of highly optimized code part

luminosities = xr.DataArray(luminosities, dims=["t", "led_name"], name="luminosity")
luminosities["t"] = np.arange(luminosities.sizes["t"])/fps
luminosities["t"].attrs["fs"] = fps
luminosities = luminosities/image["mask"].sum(["y", "x"])
luminosities

luminosities.rename(led_name="channel").to_zarr(a.output_path)