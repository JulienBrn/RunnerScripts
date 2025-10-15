from typing import ClassVar
from pydantic import Field
from script2runner import CLI
from pathlib import Path

class Args(CLI):
    """
    This script calculates prediction errors (RMSE and histograms of percentage errors)
    by comparing the model's predicted coordinates to the ground-truth annotations.

    Before running this script:
    - Run predictions with `DLC/Predictions.py` on the video `DLC_Project/videos/output_video.avi` from the DeepLabCut project 
      to generate and save the CSV file containing the predictions.

    Parameters:
    - model_DLC_path: Path to the root directory of the DeepLabCut project.
    - csv_path_ouput_video_of_DLC: Path to the CSV file with DLC predictions for `output_video.avi`.
    - output_eval_path: Directory where the evaluation results will be saved.
    """
    model_DLC_path: str = Field(..., examples=['/media/filer2/T4b/UserFolders/Réjane/Resultats_DLC/Model_monkeys/Modelconfig_monkey_0_2_10/DLC-project-2025-06-20'])
    csv_path_ouput_video_of_DLC: str = Field(..., examples=['/media/filer2/T4b/UserFolders/Réjane/testDLC_Resnet50_DLCJun20shuffle1_snapshot_010.csv'])
    output_eval_path: Path = Field(..., description="The path where the generated project will be saved.", examples=["/media/filer2/T4b/UserFolders/Réjane"]) 
    _run_info: ClassVar = dict(conda_env="dlc_py", uses_gpu=True)

a = Args()
import numpy as np, pandas as pd, xarray as xr, os, matplotlib.pyplot as plt, plotly.graph_objects as go, yaml

def calculate_rmse(df_pred, df_true):
    results = {}
    bodyparts_pred = sorted(set([col[2:] for col in df_pred.columns if col.startswith('x_')]))
    bodyparts_true = sorted(set([col[2:] for col in df_true.columns if col.startswith('x_')]))
    common_bodyparts = set(bodyparts_pred).intersection(bodyparts_true)
    if not common_bodyparts:
        print("Warning: No common bodyparts found between prediction and truth.")
        return results
    for bp in sorted(common_bodyparts):
        col_pred_x = f'x_{bp}'
        col_pred_y = f'y_{bp}'
        col_true_x = f'x_{bp}'
        col_true_y = f'y_{bp}'
        if col_pred_x in df_pred.columns and col_pred_y in df_pred.columns \
                and col_true_x in df_true.columns and col_true_y in df_true.columns:
            x_pred = df_pred[col_pred_x].to_numpy(dtype=float)
            y_pred = df_pred[col_pred_y].to_numpy(dtype=float)
            x_true = df_true[col_true_x].to_numpy(dtype=float)
            y_true = df_true[col_true_y].to_numpy(dtype=float)
            if len(x_pred) != len(x_true):
                print(f"Warning: Length mismatch for bodypart '{bp}': pred={len(x_pred)} true={len(x_true)}")
                min_len = min(len(x_pred), len(x_true))
                x_pred = x_pred[:min_len]
                y_pred = y_pred[:min_len]
                x_true = x_true[:min_len]
                y_true = y_true[:min_len]
            rmse_x = np.sqrt(np.nanmean((x_pred - x_true) ** 2))
            rmse_y = np.sqrt(np.nanmean((y_pred - y_true) ** 2))
            rmse_combined = np.sqrt((rmse_x ** 2 + rmse_y ** 2) / 2)
            results[bp] = {'rmse_x': rmse_x, 'rmse_y': rmse_y, 'rmse_combined': rmse_combined}
        else:
            print(f"Warning: Missing columns for bodypart '{bp}' in df_pred or df_true.")
    return results

df_predict = pd.read_csv(f'{a.csv_path_ouput_video_of_DLC}')
collecteddatah5 = str(Path(a.model_DLC_path) / "labeled-data" / "output_video" / "CollectedData_project.h5")
df_verite = pd.read_hdf(collecteddatah5)

# Prédictions
header = df_predict.iloc[:2]
data = df_predict.iloc[2:].reset_index(drop=True)
data.columns = pd.MultiIndex.from_arrays(header.values)
coords = data[('bodyparts', 'coords')].astype(int)
df_pred = pd.DataFrame()
df_pred['coords'] = coords
for bodypart in data.columns.levels[0]:
    if bodypart == 'bodyparts':
        continue
    if ('x' in data[bodypart]) and ('y' in data[bodypart]):
        df_pred[f'x_{bodypart}'] = data[(bodypart, 'x')].astype(float)
        df_pred[f'y_{bodypart}'] = data[(bodypart, 'y')].astype(float)

# Vérité terrain
df_true = pd.DataFrame()
df_true['coords'] = range(len(df_verite))
bodyparts = sorted(df_verite.columns.get_level_values(1).unique())
for bp in bodyparts:
    try:
        x = df_verite['project'][bp]['x']
        y = df_verite['project'][bp]['y']
    except KeyError:
        continue
    df_true[f'x_{bp}'] = x.values
    df_true[f'y_{bp}'] = y.values

# Calcul RMSE
df_pred_simple = df_pred.set_index('coords')
df_true_simple = df_true.set_index('coords')
rmse_results = calculate_rmse(df_pred_simple, df_true_simple)

# 1er enregistrement : tableau RMSE
output_dir = str(Path(a.output_eval_path) / "eval_quantitative")
os.makedirs(output_dir, exist_ok=True)
data = [{'bodypart': bp, 'rmse': vals['rmse_combined']} for bp, vals in rmse_results.items()]
df_rmse = pd.DataFrame(data)
rmse_values = [vals['rmse_combined'] for vals in rmse_results.values()]
rmse_global = np.sqrt(np.mean(np.square(rmse_values)))
bodyparts = list(df_rmse['bodypart']) + ['Global']
rmses = list(df_rmse['rmse']) + [rmse_global]
fill_colors = ['lavender'] * len(bodyparts)
fill_colors[-1] = 'lightcoral'
fig = go.Figure(data=[go.Table(
    header=dict(values=["Bodypart", "RMSE"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[bodyparts, rmses],
               fill_color=[fill_colors, fill_colors],
               align='left'))
])
fig.update_layout(title="Tableau des RMSE")
output_path = os.path.join(output_dir, "table_rmse.html")
fig.write_html(output_path)
print(f"Tableau sauvegardé dans : {output_path}")

# 2eme enregistrement : histogrammes pourcentage
df_long = df_true_simple.copy()
df_long['frame_id'] = df_long.index
if 'value' in df_long.columns:
    df_long = df_long.rename(columns={'value': 'value_old'})
df_long = df_long.melt(id_vars='frame_id', var_name='coord', value_name='value')
df_long['coord_type'] = df_long['coord'].str.extract(r'^(x|y)')
df_long['bodypart'] = df_long['coord'].str.extract(r'^[xy]_(.*)')
df_tidy = df_long.pivot(index=['frame_id', 'bodypart'], columns='coord_type', values='value').reset_index()
ds_true_simple = df_tidy.set_index(['frame_id', 'bodypart']).to_xarray()

df_long_pred = df_pred_simple.copy()
df_long_pred['frame_id'] = df_long_pred.index
if 'value' in df_long_pred.columns:
    df_long_pred = df_long_pred.rename(columns={'value': 'value_old'})
df_long_pred = df_long_pred.melt(id_vars='frame_id', var_name='coord', value_name='value')
df_long_pred['coord_type'] = df_long_pred['coord'].str.extract(r'^(x|y)')
df_long_pred['bodypart'] = df_long_pred['coord'].str.extract(r'^[xy]_(.*)')
df_tidy_pred = df_long_pred.pivot(index=['frame_id', 'bodypart'], columns='coord_type', values='value').reset_index()
ds_pred_simple = df_tidy_pred.set_index(['frame_id', 'bodypart']).to_xarray()

# Étape 1 – Ajouter une coordonnée source
ds_true_simple = ds_true_simple.expand_dims(dim='source')
ds_true_simple = ds_true_simple.assign_coords(source=['true'])
ds_pred_simple = ds_pred_simple.expand_dims(dim='source')
ds_pred_simple = ds_pred_simple.assign_coords(source=['pred'])
# Étape 2 – Concaténer le long de la nouvelle dimension 'source'
ds_both = xr.concat([ds_true_simple, ds_pred_simple], dim='source')
x_true = ds_both.sel(source='true')['x']
y_true = ds_both.sel(source='true')['y']
x_pred = ds_both.sel(source='pred')['x']
y_pred = ds_both.sel(source='pred')['y']
distance = np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
ds_dist = distance.to_dataset(name='distance')
with open(f"{a.model_DLC_path}/config.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
video_path = next(iter(data["video_sets"]))
crop_str = data["video_sets"][video_path]["crop"]
x_min, x_max, y_min, y_max = map(int, crop_str.split(","))
width = x_max - x_min
height = y_max - y_min
ds_both = ds_both.assign_attrs(image_size=(width, height))
ds_both_with_dist = ds_both.merge(ds_dist)
image_width, image_height = ds_both.attrs['image_size']
print(image_width)
image_diag = np.sqrt(image_width**2 + image_height**2)
error_percent = (ds_both_with_dist.distance / image_diag) * 100
ds_both_with_dist = ds_both_with_dist.assign(error_percent=error_percent)

df = ds_both_with_dist.to_dataframe().reset_index()
df = df.dropna(subset=['error_percent', 'bodypart'])
bodyparts = df['bodypart'].unique()
n_parts = len(bodyparts)
cols = 4
rows = (n_parts + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
axes = axes.flatten()
for i, bp in enumerate(bodyparts):
    ax = axes[i]
    errors = df[df['bodypart'] == bp]['error_percent']
    counts, bins = np.histogram(errors, bins=20)
    probs = counts / counts.sum()
    ax.bar(bins[:-1], probs, width=np.diff(bins), color='coral', edgecolor='black', align='edge')
    ax.set_title(bp)
    ax.set_xlabel("Error (%)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.tight_layout()
plt.suptitle("Histogram error (%)", fontsize=16, y=1.02)
output_img_path = os.path.join(output_dir, "histogram_distance.png")
fig.savefig(output_img_path, dpi=300)
output_path = os.path.join(output_dir, "histogram_distance.html")
with open(output_path, "w") as f:
    f.write(f"<html><body><img src='{output_img_path}'></body></html>")
print(f"Histogrammes sauvegardés dans : {output_path}")