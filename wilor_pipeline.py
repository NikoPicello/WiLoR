import os, sys
# Exclude ~/.local site-packages to avoid numpy binary incompatibility
# (packages there may be compiled against a different numpy version)
sys.path = [p for p in sys.path if '.local' not in p]

from pathlib import Path
import pickle
import torch
import argparse
import cv2 as cv
import numpy as np
import json
import glob
import imageio
from typing import Dict, Optional
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as SciR

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

cam_map = {
  'GC' : 'GB',
  'HC' : 'GF',
  'Z1' : 'FC1',
  'Z2' : 'FC2',
  'N1' : 'HA1',
  'N2' : 'HA2'
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']
vis = False
def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  main_path = '/'.join(sys.path[0].split('/')[:-2]) + '/'
  resources_path = os.path.join(main_path, 'resources')
  calibs_path   = os.path.join(resources_path, 'calibs')
  sessions_path = os.path.join(resources_path, 'sessions')
  out_path = os.path.join(resources_path, 'wilor_results')
  sid_paths = sorted(glob.glob(sessions_path + '/*'))

  # Download and load checkpoints
  model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
  detector = YOLO('./pretrained_models/detector.pt')
  # Setup the renderer
  renderer = Renderer(model_cfg, faces=model.mano.faces)
  renderer_side = Renderer(model_cfg, faces=model.mano.faces)

  device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model    = model.to(device)
  detector = detector.to(device)
  model.eval()

  # Make output directory if it does not exist
  os.makedirs(out_path, exist_ok=True)
  # Save MANO face topology once so downstream scripts can use it without the model loaded
  np.save(os.path.join(out_path, 'mano_faces.npy'), np.array(model.mano.faces))

  for sid_path in sid_paths:
    session_id = Path(sid_path).stem
    with open(os.path.join(sid_path, 'session_data.txt')) as f:
      lines = f.readlines()
      calib_date = lines[1][11:].strip()
    curr_calib_path = os.path.join(calibs_path, calib_date)
    cam_calibs = glob.glob(curr_calib_path + '/*')
    cam_dict = {}
    for cam_calib in cam_calibs:
      cam_name = Path(cam_calib).stem
      fs = cv.FileStorage(os.path.join(calibs_path, f"{calib_date}/{cam_name}.yml"), cv.FILE_STORAGE_READ)
      K = fs.getNode('K').mat()
      D = fs.getNode('D').mat()
      R = fs.getNode('R').mat()
      T = fs.getNode('T').mat()
      fs.release()
      cam_dict[cam_map[cam_name]] = {'K' : K, 'D' : D, 'R' : R, 'T' : T}

    for activity in activities:
      vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      for vid_path in vid_paths:
        video_name = Path(vid_path).stem
        K = cam_dict[video_name]['K']

        cap = cv.VideoCapture(vid_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        curr_out_path = os.path.join(out_path, f"{session_id}/{activity}")
        out_mano_path = os.path.join(curr_out_path, f"{video_name}_mano")
        out_vid_path = os.path.join(curr_out_path, f"{video_name}_render.mp4")
        out_pkl_path = os.path.join(curr_out_path, f"{video_name}_wilor.pkl")
        os.makedirs(curr_out_path, exist_ok=True)
        os.makedirs(out_mano_path, exist_ok=True)
        if vis:
          writer = imageio.get_writer(
              out_vid_path,
              fps=fps, mode='I', format='FFMPEG', macro_block_size=1
          )

        # List of per-hand detection dicts saved at the end of each video
        all_detections = []
        print(f"total frames = {total_frames}")
        # for fidx in trange(total_frames):
        for fidx in trange(5):
          ret, frame = cap.read()
          if not ret: break
          frame = cv.resize(frame, (1280, 720))




          out_frame_dict = {
            "fidx": fidx,
          }

          detections = detector(frame, conf = 0.3, verbose=False)[0]
          bboxes    = []
          is_right  = []
          for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

          if len(bboxes) == 0:
            continue
          boxes = np.stack(bboxes)
          right = np.stack(is_right)
          dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=2.0)
          dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

          all_verts = []
          all_cam_t = []
          all_right = []
          all_joints= []
          all_kpts  = []

          for batch in dataloader:
            batch = recursive_to(batch, device)

            with torch.no_grad():
              out = model(batch)
            # for k, v in out.items():
            #   if isinstance(v, dict):
            #     print(f"{k}")
            #     for k1, v1 in v.items():
            #       print(f" -{k1} : {v1.shape}")
            #   else:
            #     print(f"{k} : {v.shape}")
            # return

            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()


            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
              # Get filename from path img_path
              # img_fn, _ = os.path.splitext(os.path.basename(img_path))

              verts  = out['pred_vertices'][n].detach().cpu().numpy()
              joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

              is_right    = batch['right'][n].cpu().numpy()
              verts[:,0]  = (2*is_right-1)*verts[:,0]
              joints[:,0] = (2*is_right-1)*joints[:,0]
              cam_t = pred_cam_t_full[n]
              kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])

              h_idx = len(all_verts)  # unique per-hand index within this frame
              all_verts.append(verts)
              all_cam_t.append(cam_t)
              all_right.append(is_right)
              all_joints.append(joints)
              all_kpts.append(kpts_2d)

              # ── extract MANO params; convert rotation matrices → axis-angle ──
              sf = float(scaled_focal_length)
              go_rotmat = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy().reshape(-1, 3, 3)  # (1, 3, 3)
              hp_rotmat = out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy().reshape(-1, 3, 3)       # (15, 3, 3)
              go_aa = SciR.from_matrix(go_rotmat).as_rotvec()   # (1, 3)
              hp_aa = SciR.from_matrix(hp_rotmat).as_rotvec()   # (15, 3)
              betas_n = out['pred_mano_params']['betas'][n].detach().cpu().numpy()  # (10,)

              # ── project hand keypoints to full-image pixel space (numpy) ──
              cx = float(img_size[n][0]) / 2.0
              cy = float(img_size[n][1]) / 2.0
              pts_cam = joints + cam_t
              pts_cam = pts_cam / pts_cam[:, 2:3]
              kpts_2d_joints = np.stack([pts_cam[:, 0] * sf + cx,
                                         pts_cam[:, 1] * sf + cy], axis=1)  # (21, 2)

              hand_side = 'right' if bool(is_right) else 'left'
              all_detections.append({
                  'frame_index':      fidx,
                  'hand_side':        hand_side,
                  'kpt2d':            kpts_2d_joints,   # (21, 2) pixel coords
                  'kpt3d':            joints,           # (21, 3) camera space
                  'vertices':         verts,            # (778, 3) camera space
                  'cam_t':            cam_t,            # (3,)
                  'focal':            sf,               # float
                  'hand_pose_aa':     hp_aa,            # (15, 3)
                  'global_orient_aa': go_aa,            # (1, 3)
                  'betas':            betas_n,          # (10,)
              })

              # Keep the trimesh for optional .obj export
              camera_translation = cam_t.copy()
              tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
              tmesh.export(os.path.join(out_mano_path, f'f{fidx}_h{h_idx}.obj'))

          # Render front view
          if vis and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = frame.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            writer.append_data((255*input_img_overlay).astype(np.uint8)[:, :, ::-1])


        cap.release()
        if vis: writer.close()
        with open(out_pkl_path, 'wb') as f:
          pickle.dump(all_detections, f)
        print(f"  Saved {len(all_detections)} hand detections → {out_pkl_path}")

def project_full_img(points, cam_trans, focal_length, img_res):
  """Project 3D points (N,3) to full-image 2D pixel coords (N,2).

  points:       numpy (N, 3) in camera space
  cam_trans:    numpy (3,) full-image camera translation
  focal_length: scalar – scaled focal length in pixels
  img_res:      (width, height) or tensor with same layout
  """
  cx = float(img_res[0]) / 2.0
  cy = float(img_res[1]) / 2.0
  sf = float(focal_length)
  pts = np.asarray(points, dtype=np.float32) + np.asarray(cam_trans, dtype=np.float32)
  pts = pts / pts[:, 2:3]
  x = pts[:, 0] * sf + cx
  y = pts[:, 1] * sf + cy
  return np.stack([x, y], axis=1)

if __name__ == '__main__':
  main()
