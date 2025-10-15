from typing import ClassVar, List
from pydantic import Field, BaseModel
from script2runner import CLI
from pathlib import Path
import numpy as np

class DLC_Project(BaseModel):
    deeplabcut_project_path: Path = Field(..., description="Put the path to the DeepLabCut project. This corresponds to the path entered in the 'output_project_path' variable during training.", examples=["/media/filer2/T4b/UserFolders/Name/DLC-Project-2025-03-13"])
    output_h5_path: Path | None = Field(default=None, description="The path where H5 predictions are saved. If empty, the predictions will be computed", examples=["/media/filer2/T4b/UserFolders/Name/result-predict-h5.h5"], pattern="(.*\.h5$)|(^$)")
    videos: List[Path] = Field(..., description="List of videos to analyze with DeepLabCut.", 
                               examples=[["/media/filer2/T4b/myvideo1.mp4", "/media/filer2/T4b/myvideo2.mp4"]])

class Args(CLI):
    """
    ### **Pose Estimation with DeepLabCut**  

    #### **Objective** : The goal of this project is to perform pose estimation on videos using DeepLabCut. 
    The predicted keypoints (body parts) will be stored in an `.h5` file, allowing for visualization of the selected video with the estimated poses.  

    **Output Files** :
    1. **Velocity Analysis (`vitesse_bodyparts_video_name.html`)** : 
        - An interactive HTML file visualizing the velocity of each body part throughout the video.  
        - A well-performing model should display fewer velocity spikes, indicating stable and accurate predictions.  

    2. **Motion Heatmaps (`heatmap_mvt/`)** : 
        - A folder containing an HTML heatmap for each body part.  
        - These visualizations illustrate the predicted positions of each body part superimposed on the video frames.  

    3. **Distance Variability Heatmaps (`heatmap_distance/`)** : 
        - A folder containing an HTML file for each body part.  
        - These visualizations highlight the frame-to-frame variability in the predicted positions, helping to assess consistency and accuracy.  
    """
    deeplabcut_projet_info: DLC_Project
    output_evaluation_path: Path = Field(..., examples=["/media/filer2/T4b/UserFolders/Name"])
    _run_info: ClassVar = dict(conda_env="dlc_py")

a = Args()

import deeplabcut
import cv2
import h5py
import yaml
import matplotlib.cm as cm
import calendar
import re
import pandas as pd
import os
import plotly.graph_objects as go

# Fonction pour calculer la distance
def calculate_distance(coords_1, coords_2):
    distances = []
    coords_1 = np.array(coords_1).reshape(-1, 3)
    coords_2 = np.array(coords_2).reshape(-1, 3)  
    for i in range(len(coords_1)):
        distance = np.linalg.norm(coords_1[i, :2] - coords_2[i, :2])
        distances.append(distance)
    return distances


config_path = a.deeplabcut_projet_info.deeplabcut_project_path / 'config.yaml'
video_paths = [str(video) for video in a.deeplabcut_projet_info.videos]

if a.deeplabcut_projet_info.output_h5_path is not None:
    for video_path in video_paths:
            filename_video = Path(video_path).stem  
            output_video_path = a.output_evaluation_path / f'evaluate_{filename_video}'
            output_video_path.parent.mkdir(parents=True, exist_ok=True)

    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(a.deeplabcut_projet_info.deeplabcut_project_path))
    if match:
        year, month, day = match.groups()
        month_abbr = calendar.month_abbr[int(month)] 
        month_day = f"{month_abbr}{day}"  
    else:
        print("No date.")

    pytorch_config_path = a.deeplabcut_projet_info.deeplabcut_project_path / 'dlc-models-pytorch' / 'iteration-0' / f'DLC{month_day}-trainset95shuffle1' / 'train' / 'pytorch_config.yaml'
    with open(pytorch_config_path, 'r') as file:
        config = yaml.safe_load(file)
    num_epochs = str(config['train_settings']['epochs']).zfill(3)

    for video_path in video_paths:
        filename_video = Path(video_path).stem
        output_video_path = a.output_evaluation_path / f'evaluate_{filename_video}'
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        h5_path = a.deeplabcut_projet_info.output_h5_path

        with h5py.File(h5_path, 'r') as f:
            table = f['df_with_missing']['table']
            for frame_num, coordinates in table:
                coords = coordinates
            all_coords = [coords for _, coords in table]

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        bodyparts = config.get("bodyparts", [])
        skeleton_raw = config.get("skeleton", [])

        if isinstance(skeleton_raw, list) and all(isinstance(item, str) for item in skeleton_raw):
            print("Warning: 'skeleton' is a list of strings instead of a list of links. Fixing format.")
            skeleton = [tuple(s.split()) for s in skeleton_raw]  
        else:
            skeleton = [tuple(link) for link in skeleton_raw]

        with h5py.File(h5_path, 'r') as f:
            table = f['df_with_missing']['table']
            all_coords = [coords for _, coords in table]  

        num_bodyparts = len(bodyparts)  

        skeleton_indices = []
        for link in skeleton:
            if isinstance(link, (list, tuple)) and len(link) == 2:
                c, b = link
                if isinstance(c, str) and isinstance(b, str):
                    if c in bodyparts and b in bodyparts:
                        skeleton_indices.append((bodyparts.index(c), bodyparts.index(b)))
                    else:
                        print(f"Warning: One or both bodyparts in {link} not found in bodyparts list. Skipping.")
                else:
                    print(f"Warning: Unexpected types in link {link}. Expected (str, str), got ({type(c)}, {type(b)}). Skipping.")
            else:
                print(f"Warning: Invalid skeleton link {link}. Skipping.")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path2 = a.output_evaluation_path / f'evaluate_{filename_video}' / f'{filename_video}_predictions.avi'
        output_video_path2.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        out = cv2.VideoWriter(str(output_video_path2), fourcc, fps, (frame_width, frame_height))

        frame_idx = 0

        cmap = cm.get_cmap('jet', num_bodyparts)
        colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_bodyparts)]
        bodypart_colors = {bodyparts[i]: colors[i] for i in range(num_bodyparts)}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            coords = all_coords[frame_idx]
            points = np.array(coords).reshape(num_bodyparts, 3)[:, :2] 
            confidence = np.array(coords).reshape(num_bodyparts, 3)[:, 2]  

            for i, j in skeleton_indices:
                if (
                    not np.isnan(points[i]).any() and not np.isnan(points[j]).any()
                    and confidence[i] > 0.8 and confidence[j] > 0.8
                ):
                    cv2.line(frame, tuple(points[i].astype(int)), tuple(points[j].astype(int)), (0, 0, 0), 2)


            for idx, (x, y) in enumerate(points):
                if not np.isnan(x) and not np.isnan(y) and confidence[idx] > 0.8:
                    bodypart_name = bodyparts[idx]
                    color = bodypart_colors[bodypart_name]                
                    bgr_color = (color[2], color[1], color[0])         
                    cv2.circle(frame, (int(x), int(y)), 5, bgr_color, -1)
            out.write(frame)

            frame_idx += 1
            if frame_idx >= len(all_coords):  
                break

        cap.release()
        out.release()

        frame_distances = {bodypart: [] for bodypart in bodyparts}
        for frame_idx in range(1, len(all_coords)):
            coords_1 = all_coords[frame_idx - 1]
            coords_2 = all_coords[frame_idx]
            
            distances = calculate_distance(coords_1, coords_2)

            for i, dist in enumerate(distances):
                if i < len(bodyparts):
                    frame_distances[bodyparts[i]].append(dist)

        fig = go.Figure()
        for idx, bodypart in enumerate(bodyparts):
            color = f"rgb({bodypart_colors[bodypart][0]}, {bodypart_colors[bodypart][1]}, {bodypart_colors[bodypart][2]})"
            fig.add_trace(go.Scatter(x=np.arange(1, len(all_coords)), 
                                    y=frame_distances[bodypart],
                                    mode='lines',
                                    name=bodypart,
                                    line=dict(color=color)))

        fig.update_layout(
            title="Vitesse des bodyparts entre les frames",
            xaxis_title="Numéro de frame",
            yaxis_title="Vitesse (distance entre les frames)",
            legend_title="Bodyparts",
            template="ggplot2")
        
        output_html = output_video_path / f'vitesse_bodyparts_{filename_video}.html'
        fig.write_html(output_html)

        for idx, bodypart in enumerate(bodyparts):
            x_coords = [coords[idx * 3] for coords in all_coords if not np.isnan(coords[idx * 3])]
            y_coords = [coords[idx * 3 + 1] for coords in all_coords if not np.isnan(coords[idx * 3 + 1])]    
            H, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[100, 100], range=[[0, frame_width], [0, frame_height]])
            H_log = np.log10(H)  
            hovertext = np.full_like(H, '', dtype=object)

            for i, coords in enumerate(all_coords):
                x_idx = int(coords[idx * 3] / (frame_width / 100))
                y_idx = int(coords[idx * 3 + 1] / (frame_height / 100))  
                hovertext[y_idx, x_idx] += f'{i}, ' 

            fig = go.Figure(data=go.Heatmap(
                z=H_log.T,
                x=x_edges,
                y=y_edges[::-1],
                colorscale='amp',
                hovertemplate=(
                    'X: %{x}<br>' +
                    'Y: %{y}<br>' +
                    'Frame: %{customdata}<br>' +  
                    '<extra></extra>'
                ),
                customdata=hovertext))

            fig.update_layout(
                title=f"Heatmap de la prédiction de {bodypart}",
                xaxis_title="Position X",
                yaxis_title="Position Y",
                template="ggplot2")

            output_dir = output_video_path / "heatmap_predictions"
            os.makedirs(output_dir, exist_ok=True)
            output_html = output_dir / f'heatmap_{bodypart}_{filename_video}.html'
            fig.write_html(output_html)

        for idx, bodypart in enumerate(bodyparts):
            df = pd.DataFrame()
            df["x"] = [coords[idx * 3] for coords in all_coords if not np.isnan(coords[idx * 3])]
            df["y"] = [coords[idx * 3 + 1] for coords in all_coords if not np.isnan(coords[idx * 3 + 1])]
            df["frame_num"] = np.arange(len(df.index))
            df["d"] = np.sqrt((df["x"] - df["x"].shift(1))**2 + (df["y"] - df["y"].shift(1))**2)

            x_bins = np.linspace(df["x"].min(), df["x"].max(), 100)
            df["x_bin"] = x_bins[np.digitize(df["x"], x_bins) - 1]
            y_bins = np.linspace(df["y"].min(), df["y"].max(), 100)
            df["y_bin"] = y_bins[np.digitize(df["y"], y_bins) - 1]

            heatmap_data = df.groupby(["x_bin", "y_bin"])["d"].mean().unstack()

            fig = go.Figure(data=go.Heatmap(
                z=np.log10(heatmap_data.values).T,
                x=heatmap_data.index,
                y=heatmap_data.columns[::-1],  
                colorscale='amp'))

            fig.update_layout(
                title=f"Heatmap de la prédiction mean de {bodypart}",
                xaxis_title="Position X",
                yaxis_title="Position Y",
                template="ggplot2")

            output_dir2 = output_video_path / "heatmap_max"
            os.makedirs(output_dir2, exist_ok=True)
            output_html = output_dir2 / f'heatmap_{bodypart}_{filename_video}.html'
            fig.write_html(output_html)

else:
    for video_path in video_paths:
        filename_video = Path(video_path).stem  
        output_video_path = a.output_evaluation_path / f'evaluate_{filename_video}'
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        deeplabcut.analyze_videos(config_path, video_path, save_as_csv=True, destfolder=output_video_path)

    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(a.deeplabcut_projet_info.deeplabcut_project_path))
    if match:
        year, month, day = match.groups()
        month_abbr = calendar.month_abbr[int(month)] 
        month_day = f"{month_abbr}{day}"  
    else:
        print("No date.")

    pytorch_config_path = a.deeplabcut_projet_info.deeplabcut_project_path / 'dlc-models-pytorch' / 'iteration-0' / f'DLC{month_day}-trainset95shuffle1' / 'train' / 'pytorch_config.yaml'
    with open(pytorch_config_path, 'r') as file:
        config = yaml.safe_load(file)
    num_epochs = str(config['train_settings']['epochs']).zfill(3)

    for video_path in video_paths:
        filename_video = Path(video_path).stem
        output_video_path = a.output_evaluation_path / f'evaluate_{filename_video}'
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_video}DLC_Resnet50_DLC{month_day}shuffle1_snapshot_{num_epochs}.h5"
        h5_path = Path(f'{output_video_path}/{filename}')

        with h5py.File(h5_path, 'r') as f:
            table = f['df_with_missing']['table']
            for frame_num, coordinates in table:
                coords = coordinates
            all_coords = [coords for _, coords in table]

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        bodyparts = config.get("bodyparts", [])
        skeleton_raw = config.get("skeleton", [])

        if isinstance(skeleton_raw, list) and all(isinstance(item, str) for item in skeleton_raw):
            print("Warning: 'skeleton' is a list of strings instead of a list of links. Fixing format.")
            skeleton = [tuple(s.split()) for s in skeleton_raw]  
        else:
            skeleton = [tuple(link) for link in skeleton_raw]

        with h5py.File(h5_path, 'r') as f:
            table = f['df_with_missing']['table']
            all_coords = [coords for _, coords in table]  

        num_bodyparts = len(bodyparts)  

        skeleton_indices = []
        for link in skeleton:
            if isinstance(link, (list, tuple)) and len(link) == 2:
                c, b = link
                if isinstance(c, str) and isinstance(b, str):
                    if c in bodyparts and b in bodyparts:
                        skeleton_indices.append((bodyparts.index(c), bodyparts.index(b)))
                    else:
                        print(f"Warning: One or both bodyparts in {link} not found in bodyparts list. Skipping.")
                else:
                    print(f"Warning: Unexpected types in link {link}. Expected (str, str), got ({type(c)}, {type(b)}). Skipping.")
            else:
                print(f"Warning: Invalid skeleton link {link}. Skipping.")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path2 = a.output_evaluation_path / f'evaluate_{filename_video}' / f'{filename}_predictions.avi'
        output_video_path2.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        out = cv2.VideoWriter(str(output_video_path2), fourcc, fps, (frame_width, frame_height))

        frame_idx = 0

        cmap = cm.get_cmap('jet', num_bodyparts)
        colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_bodyparts)]
        bodypart_colors = {bodyparts[i]: colors[i] for i in range(num_bodyparts)}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            coords = all_coords[frame_idx]
            points = np.array(coords).reshape(num_bodyparts, 3)[:, :2] 
            confidence = np.array(coords).reshape(num_bodyparts, 3)[:, 2]  

            for i, j in skeleton_indices:
                if not np.isnan(points[i]).any() and not np.isnan(points[j]).any():
                    cv2.line(frame, tuple(points[i].astype(int)), tuple(points[j].astype(int)), (0, 0, 0), 2) 

            for idx, (x, y) in enumerate(points):
                if not np.isnan(x) and not np.isnan(y) and confidence[idx] > 0.0:
                    bodypart_name = bodyparts[idx]
                    color = bodypart_colors[bodypart_name]                
                    bgr_color = (color[2], color[1], color[0])         
                    cv2.circle(frame, (int(x), int(y)), 5, bgr_color, -1)
            out.write(frame)

            frame_idx += 1
            if frame_idx >= len(all_coords):  
                break

        cap.release()
        out.release()

        frame_distances = {bodypart: [] for bodypart in bodyparts}
        for frame_idx in range(1, len(all_coords)):
            coords_1 = all_coords[frame_idx - 1]
            coords_2 = all_coords[frame_idx]
            
            distances = calculate_distance(coords_1, coords_2)

            for i, dist in enumerate(distances):
                if i < len(bodyparts):
                    frame_distances[bodyparts[i]].append(dist)

        fig = go.Figure()
        
        for idx, bodypart in enumerate(bodyparts):
            color = f"rgb({bodypart_colors[bodypart][0]}, {bodypart_colors[bodypart][1]}, {bodypart_colors[bodypart][2]})"
            fig.add_trace(go.Scatter(x=np.arange(1, len(all_coords)), 
                                    y=frame_distances[bodypart],
                                    mode='lines',
                                    name=bodypart,
                                    line=dict(color=color)))

        fig.update_layout(
            title="Vitesse des bodyparts entre les frames",
            xaxis_title="Numéro de frame",
            yaxis_title="Vitesse (distance entre les frames)",
            legend_title="Bodyparts",
            template="ggplot2")
        
        output_html = output_video_path / f'vitesse_bodyparts_{filename_video}.html'
        fig.write_html(output_html)

        for idx, bodypart in enumerate(bodyparts):
            x_coords = [coords[idx * 3] for coords in all_coords if not np.isnan(coords[idx * 3])]
            y_coords = [coords[idx * 3 + 1] for coords in all_coords if not np.isnan(coords[idx * 3 + 1])]
            
            H, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[100, 100], range=[[0, frame_width], [0, frame_height]])
            H_log = np.log10(H)  

            hovertext = np.full_like(H, '', dtype=object)

            for i, coords in enumerate(all_coords):
                x_idx = int(coords[idx * 3] / (frame_width / 100))
                y_idx = int(coords[idx * 3 + 1] / (frame_height / 100))  
                hovertext[y_idx, x_idx] += f'{i}, ' 

            fig = go.Figure(data=go.Heatmap(
                z=H_log.T,
                x=x_edges,
                y=y_edges[::-1],
                colorscale='amp',
                hovertemplate=(
                    'X: %{x}<br>' +
                    'Y: %{y}<br>' +
                    'Frame: %{customdata}<br>' +  
                    '<extra></extra>'
                ),
                customdata=hovertext))

            fig.update_layout(
                title=f"Heatmap de la prédiction de {bodypart}",
                xaxis_title="Position X",
                yaxis_title="Position Y",
                template="ggplot2")

            output_dir = output_video_path / "heatmap_predictions"
            os.makedirs(output_dir, exist_ok=True)
            output_html = output_dir / f'heatmap_{bodypart}_{filename_video}.html'
            fig.write_html(output_html)

        for idx, bodypart in enumerate(bodyparts):
            df = pd.DataFrame()
            df["x"] = [coords[idx * 3] for coords in all_coords if not np.isnan(coords[idx * 3])]
            df["y"] = [coords[idx * 3 + 1] for coords in all_coords if not np.isnan(coords[idx * 3 + 1])]
            df["frame_num"] = np.arange(len(df.index))
            df["d"] = np.sqrt((df["x"] - df["x"].shift(1))**2 + (df["y"] - df["y"].shift(1))**2)

            x_bins = np.linspace(df["x"].min(), df["x"].max(), 100)
            df["x_bin"] = x_bins[np.digitize(df["x"], x_bins) - 1]
            y_bins = np.linspace(df["y"].min(), df["y"].max(), 100)
            df["y_bin"] = y_bins[np.digitize(df["y"], y_bins) - 1]

            heatmap_data = df.groupby(["x_bin", "y_bin"])["d"].mean().unstack()

            fig = go.Figure(data=go.Heatmap(
                z=np.log10(heatmap_data.values).T,
                x=heatmap_data.index,
                y=heatmap_data.columns[::-1],  
                colorscale='amp'))

            fig.update_layout(
                title=f"Heatmap de la prédiction mean de {bodypart}",
                xaxis_title="Position X",
                yaxis_title="Position Y",
                template="ggplot2")

            output_dir2 = output_video_path / "heatmap_max"
            os.makedirs(output_dir2, exist_ok=True)
            output_html = output_dir2 / f'heatmap_{bodypart}_{filename_video}.html'
            fig.write_html(output_html)
