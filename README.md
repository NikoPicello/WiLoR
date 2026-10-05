# WiLoR — hand reconstruction stage of the UBInteract 3D pipeline

This repository is a fork of **[WiLoR](https://rolpotamias.github.io/WiLoR/)** (Potamias et al., *"WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild"*, CVPR 2025), adapted to run as the **hand-reconstruction stage** of the UBInteract multi-camera 3D reconstruction pipeline.

For every session, activity and camera it:

1. detects hands in each frame with WiLoR's YOLO-based hand detector,
2. regresses a full **MANO** hand model (pose, shape, camera translation) for each detected hand,
3. saves, for each hand, the 2D keypoints, the MANO finger pose and the detector confidence (the inputs of the hand triangulation stage) to one pickle per camera,
4. optionally renders an overlay video and exports a per-hand `.obj` mesh.

Two entry points are provided:

| Script | Purpose |
|---|---|
| [`wilor_pipeline.py`](wilor_pipeline.py) | Processes one or more sessions on a single device. |
| [`run_parallel_sessions.py`](run_parallel_sessions.py) | Orchestrator: stages sessions from the dataset pool, runs `wilor_pipeline.py` on each one pinned to an idle GPU, and cleans up. |

The original WiLoR model code lives under [`wilor/`](wilor/). The only notable change is in `wilor/utils/renderer.py`, which gained a headless multi-hand renderer (`render_rgba_multiple_osmesa`).

---

## 1. Installation

```bash
git clone git@github.com:NikoPicello/WiLoR.git
cd WiLoR
```

Tested with Python 3.10, PyTorch 2.0.0 and CUDA 11.7. A conda environment is recommended:

```bash
conda create --name wilor python=3.10
conda activate wilor

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Pretrained models

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt      -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```

`pretrained_models/model_config.yaml` and `pretrained_models/dataset_config.yaml` are already tracked in the repo.

### MANO

Download the MANO model from the [MANO website](https://mano.is.tue.mpg.de) (sign up, then download `mano_v*_*.zip`). Unzip it and place the right-hand model `MANO_RIGHT.pkl` in `mano_data/` (next to the tracked `mano_mean_params.npz`). Left hands are handled by mirroring the right-hand model, so `MANO_LEFT.pkl` is not needed.

MANO is distributed under the [MANO license](https://mano.is.tue.mpg.de/license.html).

### Rendering backend (only needed for `--vis`)

`wilor/utils/renderer.py` imports `pyrender` unconditionally, so a working OpenGL backend is required **even without `--vis`**. The pipeline defaults to `PYOPENGL_PLATFORM=osmesa` (headless, CPU) if the variable is not already set; EGL can be used instead by exporting `PYOPENGL_PLATFORM=egl` before launching. With EGL, `EGL_DEVICE_ID` is automatically pinned to the first GPU in `CUDA_VISIBLE_DEVICES`, so the render context lands on the same GPU as the model.

If the import fails with `AttributeError: 'NoneType' object has no attribute 'glGetError'`, the selected backend's system library is missing (e.g. install `libosmesa6-dev` for OSMesa, or switch to EGL).

---

## 2. Data layout

Paths are resolved relative to the repository location: the scripts expect `WiLoR/` to sit two levels below the project root, next to a `resources/` folder.

```text
3D_reconstruction/
├── resources/
│   ├── all_sessions/                 # full dataset pool (read-only, used by run_parallel_sessions.py)
│   │   └── <sid>/
│   │       ├── session_data.txt
│   │       └── <activity>/<camera>.mp4
│   ├── sessions/                     # working copy actually read by wilor_pipeline.py
│   │   └── <sid>/
│   │       ├── session_data.txt
│   │       └── <activity>/
│   │           ├── <camera>.mp4                  # video mode (--use_video)
│   │           └── <camera>/000000.jpeg, ...     # image mode (default)
│   ├── calibs/
│   │   └── <calib_date>/<calib_cam>.yml          # OpenCV FileStorage with K, D, R, T
│   └── wilor_results/                # output (see §4)
└── scripts/
    ├── extract_frames.py             # mp4 → per-camera JPEG folders
    └── WiLoR/                        # this repository
```

- **`<sid>`** is a 6-digit session id (e.g. `004096`).
- **`<activity>`** is one of `animals_task`, `gaze_task`, `ghost_task`, `lego_task`, `talk_task`.
- **`session_data.txt`** must contain the calibration date on its second line, as `calib_date=YYYYMMDD`. This selects the folder under `resources/calibs/`.
- **Cameras.** Video files and calibration files use different names for the same camera. The mapping is hard-coded in `wilor_pipeline.py` (`cam_map`):

  | Calibration file | Video / frame folder |
  |---|---|
  | `GC.yml` | `GB` |
  | `HC.yml` | `GF` |
  | `Z1.yml` | `FC1` |
  | `Z2.yml` | `FC2` |
  | `N1.yml` | `HA1` |
  | `N2.yml` | `HA2` |

  Every camera that is processed must have a matching calibration file, otherwise the run stops with a `KeyError`. In video mode, the `E1`/`E2` cameras are skipped.

### Input modes

- **Image mode (default):** reads JPEG frames from `resources/sessions/<sid>/<activity>/<camera>/*.jpeg`, produced by `../extract_frames.py`. Frames are processed in filename order, so the frame index in the output matches the frame index in the source video. Unreadable frames are skipped with a warning. The preview video is written at a fixed 25 fps (the dataset capture rate).
- **Video mode (`--use_video`):** decodes `resources/sessions/<sid>/<activity>/*.mp4` directly with OpenCV, using the video's own fps.

---

## 3. Usage

### 3.1 Single run — `wilor_pipeline.py`

Run from inside the `WiLoR/` directory (model paths are relative to it):

```bash
# all activities of session 004096, image mode
python wilor_pipeline.py --sid 004096

# two activities, first 500 frames only, render preview video + meshes
python wilor_pipeline.py --sid 004096 --activities lego_task talk_task --max-frames 500 --vis

# read mp4 files directly, smaller batch
python wilor_pipeline.py --sid 004096 --use_video -b 32
```

| Flag | Default | Description |
|---|---|---|
| `--sid` | *all sessions* | Session to process. Matched as a **substring** of the session folder name; if omitted, every folder in `resources/sessions/` is processed. |
| `--activities` | all five | One or more activity folders to process. |
| `-b`, `--batch_size` | `128` | Number of **frames** buffered per detector/WiLoR forward pass. All hands found in the buffered frames go through WiLoR in a single batch. Lower this if you run out of GPU memory. |
| `--max-frames` | `-1` (all) | Process only the first *N* frames of each camera. |
| `--use_video` | off | Read `*.mp4` instead of pre-extracted JPEG folders. |
| `--vis` | off | Also write an overlay video and per-hand `.obj` meshes (slow: rendering is done per frame). |

### 3.2 Multi-GPU batch run — `run_parallel_sessions.py`

Runs `wilor_pipeline.py` over many sessions, one session per **idle** GPU. Launch it from the already-activated `wilor` environment; each child process uses the same Python interpreter.

For each session it:

1. copies `resources/all_sessions/<sid>` → `resources/sessions/<sid>` (atomic: copied to `<sid>.tmp` and then renamed, so an interrupted copy is never mistaken for a complete one; an already-staged session is not copied again);
2. runs `wilor_pipeline.py --sid <sid> [forwarded args]` with `CUDA_VISIBLE_DEVICES` set to one GPU;
3. on success, deletes `resources/sessions/<sid>` to free scratch space. On failure the staged copy is kept so a re-run can reuse it.

A GPU counts as **free** only if `nvidia-smi --query-compute-apps` reports no process on it (from any user or container), not just low utilisation. GPUs already given to one of our jobs are tracked in-process, so a job that is still starting up cannot be double-booked. A new session starts as soon as a GPU frees up.

**Note:** this pipeline reads `resources/sessions/<sid>`. Staging copies the raw session folder (mp4 files). In the default image mode, the per-camera JPEG folders must exist there, so either run `extract_frames.py` first or pass `--use-video`.

```bash
python run_parallel_sessions.py                          # auto-discover from all_sessions, keep watching forever
python run_parallel_sessions.py 000000 004096            # only these sessions, then exit
python run_parallel_sessions.py --gpus 0,1,2,3           # restrict to a GPU whitelist
python run_parallel_sessions.py 004096 --use-video --wilor-args --activities lego_task --vis
python run_parallel_sessions.py --dry-run                # log the planned actions without executing
```

| Flag | Default | Description |
|---|---|---|
| `sessions` (positional) | *auto-discover* | Explicit session ids. Without them, the script scans `resources/all_sessions/` and keeps re-scanning for new sessions (stop with Ctrl-C). Sessions that already have a non-empty `resources/wilor_results/<sid>/` are skipped. |
| `--gpus` | all GPUs (or `$GPUS`) | Comma-separated whitelist of GPU indices. GPUs on the list are still used only when idle. |
| `--no-gpu-check` | off | Use the `--gpus` list as-is and never call `nvidia-smi` (useful when the probe stalls under load). Requires `--gpus`. |
| `--use-video` | off | Forwards `--use_video` to the pipeline. |
| `--launch-stagger` | `15` s | Delay between consecutive launches. GPU persistence mode is off on the cluster, so simultaneous cold starts can fail CUDA initialisation and silently fall back to CPU. `0` disables the delay. |
| `--poll-interval` | `15` s | Interval between GPU/candidate re-checks. |
| `--dry-run` | off | Log the copy/run/remove actions without executing them. |
| `--wilor-args ...` | — | Everything after this flag is forwarded to `wilor_pipeline.py`. **Must be the last flag**: anything after it (including session ids) is forwarded too. |

The first Ctrl-C stops new launches and waits for running sessions to finish. A second Ctrl-C exits immediately, leaving the running jobs and their staged data in place.

**Logs:** `run_logs/<YYYYMMDD_HHMMSS>/<sid>.log` (full stdout/stderr of each session) and `run_logs/<run_id>/summary.tsv` (`<sid>\t<exit_code>` per line).

---

## 4. Method and outputs

### 4.1 Processing steps

For each camera stream, frames are read in chunks of `--batch_size` and processed as follows:

1. **Hand detection.** The WiLoR YOLO detector (`detector.pt`, Ultralytics 8.1.34) runs on all buffered frames with confidence threshold **0.3**. It outputs a box, a left/right label and a confidence score for each hand.
2. **Cropping.** Each box is enlarged by a factor of **2.0** around its centre, cropped and resized to **256×256** (`ViTDetDataset`). Left-hand crops are flipped horizontally so that the right-hand MANO model can be used for both hands.
3. **MANO regression.** One WiLoR forward pass (ViT backbone + refinement head) over all hand crops of the chunk predicts MANO global orientation, 15 joint rotations, 10 shape coefficients and a weak-perspective camera for the crop.
4. **Back to full-image coordinates.** For left hands, the predictions are mirrored back (x of vertices/joints and the camera's x-translation are negated). The crop camera is converted to a full-image translation `cam_t` (`cam_crop_to_full`), using WiLoR's virtual pinhole camera:
   - focal length `f = 5000 / 256 · max(W, H)` (= 25000 px for 1280×720 frames),
   - principal point at the image centre `(W/2, H/2)`.
5. **2D keypoints.** The 21 3D joints are moved into the camera frame (`kpt3d + cam_t`) and projected with that virtual camera to get `kpt2d`.

> **Note on camera models:** the per-camera calibration (`K`, `D`, `R`, `T`) is loaded from `resources/calibs/`, but it is **not** used in the reconstruction. `kpt2d` is placed in the image with WiLoR's virtual camera described above (very large focal length, no distortion). Because `cam_crop_to_full` makes the projection line up with the detected hand, `kpt2d` can be used as an ordinary image-space measurement. Metric multi-view fusion is done downstream by the triangulation stage, using the calibration.

### 4.2 Output files

```text
resources/wilor_results/<sid>/<activity>/
├── <camera>_wilor.pkl              # always
├── <camera>_render.mp4             # --vis only: frames with rendered hand meshes overlaid
└── <camera>_mano/f<frame>_h<k>.obj # --vis only: mesh of the k-th hand in frame <frame>, in camera coordinates
```

Each `<camera>_wilor.pkl` is a Python `list` with **one `dict` per detected hand** (frames with no detections contribute no entries):

| Key | Type / shape | Description | Used by |
|---|---|---|---|
| `frame_index` | `int` | 0-based frame index in the source video. | triangulation: groups detections by frame |
| `hand_side` | `str` | `'left'` or `'right'` (detector label). | triangulation: handedness filter |
| `det_conf` | `float` | YOLO box confidence in `[0.3, 1]` (detections below 0.3 are discarded). | not yet read by later stages |
| `kpt2d` | `(21, 2)` float | Joint pixel coordinates in the full image (§4.1). | triangulation: wrist matching to the SMPLer-X body, multi-view DLT input, reprojection error |
| `hand_pose_aa` | `(15, 3)` float | MANO finger joint rotations, axis-angle. | triangulation: fused across views → `hand_pose` → SMPLify-X hand initialisation and pose prior |

The joints follow the **OpenPose hand order**: `0` wrist; `1–4` thumb; `5–8` index; `9–12` middle; `13–16` ring; `17–20` little finger (in each finger, from the base to the fingertip).

For left hands, `kpt2d` is already in real left-hand image coordinates. `hand_pose_aa`, however, is the raw prediction for the **flipped crop**, i.e. it describes a right-hand MANO model. `mano_triangulation.py` converts it to the left-hand convention by negating the y and z components of each axis-angle.

WiLoR also predicts 3D joints, mesh vertices, the camera translation, the global orientation and the shape coefficients. These are not saved, because no later stage uses them: the 3D hand joints come from multi-view triangulation, and the wrist orientation and shape come from the body fit. With `--vis`, the meshes are still exported as `.obj` files for visual inspection.

Reading the output:

```python
import pickle
dets = pickle.load(open('resources/wilor_results/004096/lego_task/GB_wilor.pkl', 'rb'))
right_hands_frame0 = [d for d in dets if d['frame_index'] == 0 and d['hand_side'] == 'right']
wrist_px = right_hands_frame0[0]['kpt2d'][0]
```

> Results generated before this format change also contain `kpt3d`, `vertices`, `cam_t`, `focal`, `global_orient_aa` and `betas`, and lack `det_conf`. The triangulation stage reads only the shared keys, so both formats work with it.

### 4.3 Limitations

- **No tracking or identity.** Each frame is processed independently. Hands are not linked across frames or cameras, and two hands with the same `hand_side` can appear in one frame (e.g. two people, or a false positive). Detections are kept in detector order.
- **No temporal smoothing.** The outputs can jitter from frame to frame.
- **Flattened perspective.** `kpt2d` is projected through a 25000 px virtual camera, so it is close to orthographic: foreshortening within the hand is weaker than in the real lens (see §4.1).
- **No per-keypoint score.** `det_conf` scores the whole box; WiLoR gives no confidence for individual joints.
- **Fixed thresholds.** The detector confidence (0.3) and box rescale factor (2.0) are hard-coded in `wilor_pipeline.py`.

---

## 5. Citation

```bibtex
@inproceedings{potamias2025wilor,
  title     = {{WiLoR}: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
  author    = {Potamias, Rolandos Alexandros and Zhang, Jinglei and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
