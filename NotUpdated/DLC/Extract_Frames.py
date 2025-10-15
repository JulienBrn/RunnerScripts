from typing import List, Dict, Literal, Optional, Any, ClassVar
from pydantic import Field, BaseModel, ConfigDict
from script2runner import CLI
from pathlib import Path
import yaml
import json

projects = []
labelling_path = Path("/media/filer2/T4b/Labeling")
for p in labelling_path.iterdir():
    if p.is_dir() and (p / "project_info.yaml").exists():
        with (p / "project_info.yaml").open("r") as f:
            info = yaml.safe_load(f)
        projects.append(dict(name=p.name, path=p, info=info))

class ProjectInfo(BaseModel):
    labeling_project_name: str = Field(..., description="Labeling project to which to add those images. Project must already exist.", json_schema_extra=dict(enum=[i["name"] for i in projects]))
    metadata: Dict[str, Any] = Field(..., 
                                     description="Additional metadata for the images, may use variables {parent_i} and {filename} and {frame_num}. "+
                                     "Note that the metadata keys frame_num and source_filepath are automatically added",
                              )
    
    model_config = ConfigDict(json_schema_extra=dict(allOf=[
        {"if": dict(properties=dict(labeling_project_name={"const": i["name"]})),
         "then": dict(properties=dict(metadata=dict(properties=i["info"]["required_metadata_fields"], required=list(i["info"]["required_metadata_fields"].keys())))),
         }            
    for i in projects]))

class Args(CLI):
    """
         - **Goal** : Automate the extraction of frames from videos and generate metadata in a structured **json** format. This pipeline ensures consistency and compatibility 
         with labeling tools such as Label Studio.  
         - **Technique** :
            1. Video Processing : with `uniform` or `phash` method,
            2. **json** metadata generation.
        - **Formats** :  
            - Input :  
                - Videos : `.mp4`, `.avi`,
                - Metadata configuration : `project_info.yaml`.
            - Output :  
                - Extracted frames : `.png`, 
                - Metadata : `frames_metadata.json` (structured for Label Studio).
        - **Next step** : The extracted images and metadata can now be exported to ***Label Studio*** for labeling. Then used in the “Train.py” script to train the model.
    """
    videos: List[Path] = Field(..., description="This dictionary represents all of the videos that we wish to use in the DeepLabCut project.", 
                               examples=[["/media/filer2/T4b/myvideo1.mp4", "/media/filer2/T4b/myvideo2.mp4"]])
    num_frames_per_video: int = Field(default=25, description="This integer corresponds to the number of frames that we want to extract from each video.")
    extract_method: Literal['uniform', 'phash'] = Field(default='uniform')
    random_state: Optional[int] = Field(default=42, description="If you use kmeans method. Else null.")
    labeling_info: ProjectInfo
    _run_info: ClassVar = dict(conda_env="dlc_py")

a = Args()

import imagehash
from PIL import Image
import cv2

# uniform method
def extract_frames_uniform(video_path: Path, output_dir, num_frames: int, metadata: Dict[str, Any]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames / num_frames
    global_id_counter = len(json_data) 

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read() 
        if not ret:
            break

        parent_dict = {f"parent_{j+1}": p.name for j, p in enumerate(video_path.parents)}
        metadata_filled = {key: (value.format(frame_num=i+1, **parent_dict) if isinstance(value, str) else value) for key, value in metadata.items()}
        filename_parts = [video_path.stem, f"uniform_frame_{i+1}"]
        filename_parts += [f"{value}" for value in metadata_filled.values()]
        new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + "_" + f"img{i+1}.png"
        frame_filename = output_dir / new_frame_name

        counter = 1
        frame_filename_new = frame_filename
        while frame_filename.exists():
            new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + "_" + f"img{i+1}{counter}.png"
            frame_filename_new = Path(f"{a.labeling_info.labeling_project_name}/Images/{new_frame_name}")
            frame_filename = output_dir / new_frame_name
            counter += 1

        cv2.imwrite(str(frame_filename), frame)
        label_studio_base_path = 'http://10.24.12.180:8083/ls/'
        frame_data = {
            "id": global_id_counter + 1,
            "data": {
                "frame_num": f"{i+1}", 
                "rel_img_path": str(frame_filename.relative_to(output_dir.parent)).replace('#', '%23'),
                "label_studio_img_path": f"{label_studio_base_path}{str(frame_filename_new).replace('#', '%23')}",
                "source_video_filepath": str(video_path.resolve()),
            }}
        frame_data["data"].update(metadata_filled)
        json_data.append(frame_data)
        global_id_counter += 1  
    cap.release()

# phash method
def phash_distance(frame1, frame2):
    pil1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    h1 = imagehash.phash(pil1)
    h2 = imagehash.phash(pil2)
    return h1 - h2

def extract_frames_phash(video_path: Path, output_dir: Path, max_frames: int, step: int, phash_threshold: int, metadata: Dict[str, Any]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Problem for : {video_path}")

    selected = []
    prev_frame = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    global_id_counter = len(json_data)
    while len(selected) < max_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if prev_frame is None or phash_distance(prev_frame, frame) > phash_threshold:
            i = len(selected)
            parent_dict = {f"parent_{j+1}": p.name for j, p in enumerate(video_path.parents)}
            metadata_filled = {key: (value.format(frame_num=i+1, **parent_dict) if isinstance(value, str) else value) for key, value in metadata.items()}
            new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + f"_img{i+1}.png"
            frame_filename = output_dir / new_frame_name
            counter = 1
            while frame_filename.exists():
                new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + f"_img{i+1}_{counter}.png"
                frame_filename = output_dir / new_frame_name
                counter += 1

            cv2.imwrite(str(frame_filename), frame)
            label_studio_base_path = 'http://10.24.12.180:8083/ls/'
            frame_data = {
                "id": global_id_counter + 1,
                "data": {
                    "frame_num": f"{i+1}",
                    "rel_img_path": str(frame_filename.relative_to(output_dir.parent)).replace('#', '%23'),
                    "label_studio_img_path": f"{label_studio_base_path}{a.labeling_info.labeling_project_name}/Images/{new_frame_name}",
                    "source_video_filepath": str(video_path.resolve())
                    }}
            frame_data["data"].update(metadata_filled)
            json_data.append(frame_data)
            selected.append(frame)
            prev_frame = frame
            global_id_counter += 1
        idx += step
    cap.release()

output_dir = Path(f"/media/filer2/T4b/Labeling/{a.labeling_info.labeling_project_name}/Images")
metadata = a.labeling_info.metadata
json_data = []
if a.extract_method == 'uniform':
    for video in a.videos:
        extract_frames_uniform(video, output_dir, a.num_frames_per_video, metadata)
        print('Extracted frames with uniform method.')
elif a.extract_method == 'phash':
    for video in a.videos:
        extract_frames_phash(video, output_dir, max_frames=a.num_frames_per_video, step=10, phash_threshold=10, metadata=metadata)
        print('Extracted frames with phash method.')
else:
    raise Exception("Only 'uniform' and 'phash' method are implemented.")

json_dir = labelling_path / a.labeling_info.labeling_project_name
json_output_path = json_dir / 'frames_metadata.json'
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)