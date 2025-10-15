from typing import ClassVar
from pydantic import Field, BaseModel
from script2runner import CLI
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

optimizers = ['Adam', 'AdamW', 'SGD']

class TrainParams(BaseModel):
    n_epochs: int = Field(default=500, description="Specifies the number of epochs for training the model.")
    batch_size: int = Field(default=8, description="Defines the number of training samples processed in one forward and backward pass.")
    optimizer: str = Field(default='AdamW', description="Specifies the optimization algorithm used to update the model parameters during training.", json_schema_extra=dict(enum=optimizers))

class Args(CLI):
    """
        - **Goal** : Automate the entire pipeline for creating and training a DeepLabCut (DLC) project, from video preprocessing to model training.
        - **Technique** : 
            - Extract frames from videos,
            - Transform CSV annotations into DeepLabCut-compatible formats,
            - Automatically generate the DLC project.
        - **Formats** :
            - Input: 
                - Videos: `.mp4`, `.avi`, etc...,
                - Annotations: `.csv` (containing labeled keypoints).  
            - Output: 
                - Extracted Frames: `.png`,
                - Processed Annotations: `.h5`, `.yaml`,
                - Trained Model: `.pt` (DeepLabCut model weights).  
        - **Next step** : Use the `Predictions.py` script to apply the trained model to new videos.
    """
    project_label_studio: str = Field(..., description="This is the name of your Model in the T4b/Labeling folder.", examples=['Model_Name'])
    info_skeleton: Path = Field(..., description=".yaml file with bodyparts and skeleton.", examples=['/media/filer2/T4b/UserFolders/Name/info_skeleton.yaml'])
    output_project_path: Path = Field(..., description="The path where the generated project will be saved.", examples=["/media/filer2/T4b/UserFolders/Name"])
    num_frames_for_train: int = Field(default=50, description="The number of frames to be used for training the model. The default is 50, but you can adjust this depending on the size of your dataset.")  
    images_folder: Path = Field(..., examples=["/media/filer2/T4b/Labeling/Model_Name/Images"])
    train_params: TrainParams = TrainParams()
    _run_info: ClassVar = dict(conda_env="dlc_py", uses_gpu=True)

a = Args()
import shutil
shutil.copy = shutil.copyfile
import yaml, os, json, cv2, csv, timeit
import deeplabcut

def create_video_from_frames(input_dir: Path, output_video_path: Path, fps: int = 1, num: int = None, frame_map: dict = None):
    input_dir = Path(input_dir)
    # ordered_frame_names = [frame_map[str(i)] for i in range(min(len(frame_map), num or len(frame_map)))]
    if frame_map:
        ordered_keys = sorted(frame_map.keys(), key=lambda x: int(Path(x).stem))
        ordered_frame_names = [frame_map[k] for k in ordered_keys[:num]]
    else:
        all_images = sorted(input_dir.glob("*.png"), key=lambda x: x.name)
        ordered_frame_names = [p.name for p in all_images[:num]]
    images = [input_dir / fname for fname in ordered_frame_names]
    valid_frames = [(cv2.imread(str(p)), n) for p, n in zip(images, ordered_frame_names) if cv2.imread(str(p)) is not None]
    if not valid_frames: return
    h, w, _ = valid_frames[0][0].shape
    out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for frame, _ in valid_frames:
        out.write(frame)
    out.release()

def extract_exact_frames_from_video(video_path: Path, output_dir: Path, frame_map: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    for i in range(len(frame_map)):
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(output_dir / frame_map.get(str(i), f"{i}.png")), frame)
    cap.release()

def csv_to_h5(csv_path: str):
    csv = pd.read_csv(csv_path)
    if csv.empty or 'bodypart' not in csv.columns or 'frame_id' not in csv.columns:
        return pd.DataFrame()
    scorer = csv.get('labeller', ['default'])[0]
    bodyparts = csv["bodypart"].unique()
    images = sorted(csv['frame_id'].unique())
    data = {}
    for bp in bodyparts:
        for img in images:
            subset = csv[(csv['frame_id'] == img) & (csv['bodypart'] == bp)]
            x, y = (subset['x'].values[0], subset['y'].values[0]) if not subset.empty else (None, None)
            data.setdefault((scorer, bp, 'x'), []).append(x)
            data.setdefault((scorer, bp, 'y'), []).append(y)
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys(), names=['scorer', 'bodyparts', 'coords']),
                        index=pd.MultiIndex.from_tuples([("labeled-data", "output_video", i) for i in images]))

# Annotations
metadata_annotations = Path(f"/media/filer2/T4b/Labeling/{a.project_label_studio}/Annotations")
json_list_path = a.output_project_path / "annotations_list.json"
all_annotations, frame_map = [], {}

txt_files = sorted(os.listdir(metadata_annotations))[:a.num_frames_for_train]
for i, fname in enumerate(txt_files):
    with open(metadata_annotations / fname, "r", encoding="utf-8") as f:
        annot = json.load(f)
    all_annotations.append(annot)
    try:
        rel_img = annot['task']['data']['rel_img_path']
    except KeyError as e:
        raise ValueError(f"Problème rel_img_path : {e} pour {fname}")
    # rel_img = annot.get('task', {}).get('data', {}).get('rel_img_path')
    if rel_img:
        frame_name = os.path.basename(rel_img)
        frame_map[str(i)] = frame_name
        full_img = Path("/media/filer2/T4b/Labeling") / a.project_label_studio / rel_img
        target_img = a.images_folder / frame_name
        if not target_img.exists() and full_img.exists():
            shutil.copy(full_img, target_img)

with open(json_list_path, 'w', encoding='utf-8') as f:
    json.dump(all_annotations, f, indent=4, ensure_ascii=False)

# Création vidéo
new_video_path = Path("/media/filer2/T4b/Temporary/output_video.avi")
create_video_from_frames(a.images_folder, new_video_path, fps=1, num=a.num_frames_for_train, frame_map=frame_map)

# Création du projet DeepLabCut
working_directory = a.output_project_path
deeplabcut.create_new_project('DLC', 'project', [str(new_video_path)], working_directory, copy_videos=True, multianimal=False)
date_str = datetime.today().strftime('%Y-%m-%d')
config_path = working_directory / f"DLC-project-{date_str}" / "config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config.update({
    'numframes2extract': a.num_frames_for_train,
    'numframes2pick': a.num_frames_for_train,
    'pcutoff': 0.0,
    'dotsize': 5,
    'batch_size': a.train_params.batch_size,
    'engine': 'pytorch'
})
with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)

# Extraction des images
frames_dir = working_directory / f"DLC-project-{date_str}" / "labeled-data" / "output_video"
extract_exact_frames_from_video(new_video_path, frames_dir, frame_map)


# Mise à jour du config.yaml
with open(a.info_skeleton, "r") as f:
    info_bs = yaml.safe_load(f)
original_skeleton = info_bs.get('skeleton', [])
annotated_bodyparts = set()
for annot in all_annotations:
    for result in annot.get('result', []):
        if result.get('type') == 'keypointlabels':
            annotated_bodyparts.update(result.get('value', {}).get('keypointlabels', []))
config['bodyparts'] = sorted(annotated_bodyparts)
config['skeleton'] = [
    link for link in original_skeleton
    if isinstance(link, list) and len(link) == 2 and link[0] in annotated_bodyparts and link[1] in annotated_bodyparts
]
with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)

# Conversion des annotations en .csv
csv_path = working_directory / f"DLC-project-{date_str}" / "output.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['labeller', 'video_reference', 'frame_id', 'bodypart', 'x', 'y', 'confidence'])
    for annotation in all_annotations:
        task_data = annotation.get('task', {}).get('data', {})
        video_ref = os.path.basename(os.path.dirname(task_data.get('source_video_filepath', '')))
        frame_id = os.path.basename(task_data.get('rel_img_path', f"{task_data.get('frame_num', 0)}.png"))
        for result in annotation.get('result', []):
            if result.get('type') != 'keypointlabels':
                continue
            value = result.get('value', {})
            x = value.get('x', np.nan)
            y = value.get('y', np.nan)
            ow = result.get('original_width', 1)
            oh = result.get('original_height', 1)
            x, y = (x / 100 * ow), (y / 100 * oh)
            bp = value.get('keypointlabels', [''])[0]
            conf = 1 if not np.isnan(x) and not np.isnan(y) else np.nan
            writer.writerow(['project', video_ref, frame_id, bp, x, y, conf])

# Conversion .csv to .h5
h5 = csv_to_h5(str(csv_path))
h5.to_hdf(str(frames_dir / "CollectedData_project.h5"), "keypoints")

# Dataset d'entraînement
deeplabcut.create_training_dataset(str(config_path))

# Modification de pytorch_config.yaml
month_day = datetime.today().strftime("%b") + str(datetime.today().day)
pytorch_config_path = working_directory / f"DLC-project-{date_str}" / "dlc-models-pytorch" / "iteration-0" / f"DLC{month_day}-trainset95shuffle1" / "train" / "pytorch_config.yaml"

with open(pytorch_config_path, 'r') as f:
    config_py = yaml.safe_load(f)
config_py['runner']['optimizer']['type'] = a.train_params.optimizer
config_py['train_settings']['epochs'] = a.train_params.n_epochs
config_py['snapshot'] = 50
with open(pytorch_config_path, 'w') as f:
    yaml.safe_dump(config_py, f)

# Entraînement
start = timeit.default_timer()
deeplabcut.train_network(str(config_path), shuffle=1)
stop = timeit.default_timer()
print(f"⏱️ Temps d’entraînement = {stop - start:.2f} secondes")

# Nettoyage 
def get_model_directory(base_path, date_str, month_day):
    iteration = 1
    while True:
        model_dir = base_path / f"DLC-project-{date_str}" / "dlc-models-pytorch" / "iteration-0" / f"DLC{month_day}-trainset95shuffle{iteration}" / "train"
        if not os.path.exists(model_dir):  
            os.makedirs(model_dir, exist_ok=True) 
            return model_dir 
        iteration += 1 

model_dir_path = get_model_directory(working_directory, date_str, month_day)
print(f"Répertoire pour le modèle : {model_dir_path}")

if os.path.exists(json_list_path):
    os.remove(json_list_path)
    print(f"{json_list_path} successfully deleted.")
if os.path.exists(new_video_path):
    os.remove(new_video_path)
    print(f"{new_video_path} successfully deleted.")
done_path = os.path.join(a.output_project_path, "done.txt")
with open(done_path, "w") as f:
    f.write("done")