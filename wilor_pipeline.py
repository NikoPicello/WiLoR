import os, sys
# Exclude ~/.local site-packages to avoid numpy binary incompatibility
# (packages there may be compiled against a different numpy version)
sys.path = [p for p in sys.path if '.local' not in p]
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from pathlib import Path
import pickle
import torch
import cv2 as cv
import numpy as np
import glob
import imageio
import argparse
from tqdm import trange
from scipy.spatial.transform import Rotation as SciR
from torch.utils.data import default_collate

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

cam_map = {
  'GC' : 'GB',
  'HC' : 'GF',
  'Z1' : 'FC1',
  'Z2' : 'FC2',
  'N1' : 'HA1',
  'N2' : 'HA2'
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--batch_size', type=int, default=8,
                      help="Spcecify the batch size [defualt=8]")
  parser.add_argument('--vis', action='store_true',
                      help="If set, the function enerates the video of the detected hands")
  parser.add_argument('--sid', type=str, default=None,
                      help="Specify a session to process")
  parser.add_argument('--aid', default='all', choices=['animals', 'gaze', 'ghost', 'lego', 'talk'],
                      help="Specify an activity to process among [animals|gaze|ghost|lego|talk]")
  parser.add_argument('--max_frames', type=int, default=-1,
                      help="Max number of frames being processed")
  args = parser.parse_args()
  activities = activities if args.aid in (None, 'all') else [args.aid]
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  main_path = '/'.join(sys.path[0].split('/')[:-2]) + '/'
  resources_path = os.path.join(main_path, 'resources')
  calibs_path   = os.path.join(resources_path, 'calibs')
  sessions_path = os.path.join(resources_path, 'sessions')
  out_path      = os.path.join(resources_path, 'wilor_results')
  sid_paths     = sorted(glob.glob(sessions_path + '/*'))

  model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt', cfg_path='./pretrained_models/model_config.yaml')
  detector = YOLO('./pretrained_models/detector.pt')
  if args.vis:
    renderer = Renderer(model_cfg, faces=model.mano.faces)

  model    = model.to(device)
  detector = detector.to(device)
  model.eval()

  os.makedirs(out_path, exist_ok=True)
  np.save(os.path.join(out_path, 'mano_faces.npy'), np.array(model.mano.faces))

  for sid_path in sid_paths:
    session_id = Path(sid_path).stem
    if args.sid is not None and args.sid not in session_id: continue

    with open(os.path.join(sid_path, 'session_data.txt')) as f:
      lines = f.readlines()
      calib_date = lines[1][11:].strip()
    cam_calibs = glob.glob(os.path.join(calibs_path, calib_date) + '/*')
    cam_dict = {}
    for cam_calib in cam_calibs:
      cam_name = Path(cam_calib).stem
      fs = cv.FileStorage(os.path.join(calibs_path, f"{calib_date}/{cam_name}.yml"), cv.FILE_STORAGE_READ)
      K = fs.getNode('K').mat()
      D = fs.getNode('D').mat()
      R = fs.getNode('R').mat()
      T = fs.getNode('T').mat()
      fs.release()
      cam_dict[cam_map[cam_name]] = {'K': K, 'D': D, 'R': R, 'T': T}

    for activity in activities:
      vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      for vid_path in vid_paths:
        video_name = Path(vid_path).stem
        K = cam_dict[video_name]['K']

        cap = cv.VideoCapture(vid_path)
        fps          = int(cap.get(cv.CAP_PROP_FPS))
        if args.max_frames == -1:
          total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        else:
          total_frames = args.max_frames

        curr_out_path = os.path.join(out_path, f"{session_id}/{activity}")
        out_pkl_path  = os.path.join(curr_out_path, f"{video_name}_wilor.pkl")
        os.makedirs(curr_out_path, exist_ok=True)
        if args.vis:
          out_mano_path = os.path.join(curr_out_path, f"{video_name}_mano")
          out_vid_path  = os.path.join(curr_out_path, f"{video_name}_render.mp4")
          os.makedirs(out_mano_path, exist_ok=True)
          writer = imageio.get_writer(out_vid_path, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)

        all_detections = []
        frames_buf = []
        fidxs_buf  = []

        def flush():
          if not frames_buf:
            return

          # ── YOLO on all buffered frames at once ──────────────────────────
          yolo_out = detector(frames_buf, conf=0.3, verbose=False)

          # ── build one hand-sample list across all frames ─────────────────
          samples     = []
          sample_fidx = []  # original frame index per sample
          sample_fi   = []  # position in frames_buf per sample

          for fi, (frame, dets) in enumerate(zip(frames_buf, yolo_out)):
            bboxes, rights = [], []
            for det in dets:
              Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
              rights.append(det.boxes.cls.cpu().detach().squeeze().item())
              bboxes.append(Bbox[:4].tolist())
            if not bboxes:
              continue
            dataset = ViTDetDataset(model_cfg, frame, np.stack(bboxes), np.stack(rights), rescale_factor=2.0)
            for i in range(len(dataset)):
              samples.append(dataset[i])
              sample_fidx.append(fidxs_buf[fi])
              sample_fi.append(fi)

          # ── single WiLoR forward pass for all hands ───────────────────────
          if samples:
            batch = default_collate(samples)
            batch = recursive_to(batch, device)
            with torch.no_grad():
              out = model(batch)

            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            sf = float(scaled_focal_length)

            if args.vis:
              vis_per_frame = {}  # fi -> {verts, cam_t, right, img_size}

            for n in range(len(samples)):
              fidx       = sample_fidx[n]
              fi         = sample_fi[n]
              verts      = out['pred_vertices'][n].detach().cpu().numpy()
              joints     = out['pred_keypoints_3d'][n].detach().cpu().numpy()
              is_right_n = batch['right'][n].cpu().numpy()
              verts[:,0]  = (2*is_right_n-1)*verts[:,0]
              joints[:,0] = (2*is_right_n-1)*joints[:,0]
              cam_t = pred_cam_t_full[n]

              go_rotmat = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy().reshape(-1, 3, 3)
              hp_rotmat = out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy().reshape(-1, 3, 3)
              go_aa     = SciR.from_matrix(go_rotmat).as_rotvec()
              hp_aa     = SciR.from_matrix(hp_rotmat).as_rotvec()
              betas_n   = out['pred_mano_params']['betas'][n].detach().cpu().numpy()

              cx = float(img_size[n][0]) / 2.0
              cy = float(img_size[n][1]) / 2.0
              pts_cam = joints + cam_t
              pts_cam = pts_cam / pts_cam[:, 2:3]
              kpts_2d_joints = np.stack([pts_cam[:, 0] * sf + cx,
                                         pts_cam[:, 1] * sf + cy], axis=1)

              all_detections.append({
                  'frame_index':      fidx,
                  'hand_side':        'right' if bool(is_right_n) else 'left',
                  'kpt2d':            kpts_2d_joints,
                  'kpt3d':            joints,
                  'vertices':         verts,
                  'cam_t':            cam_t,
                  'focal':            sf,
                  'hand_pose_aa':     hp_aa,
                  'global_orient_aa': go_aa,
                  'betas':            betas_n,
              })

              if args.vis:
                if fi not in vis_per_frame:
                  vis_per_frame[fi] = {'verts': [], 'cam_t': [], 'right': [], 'img_size': img_size[n]}
                h_idx = len(vis_per_frame[fi]['verts'])
                vis_per_frame[fi]['verts'].append(verts)
                vis_per_frame[fi]['cam_t'].append(cam_t)
                vis_per_frame[fi]['right'].append(is_right_n)
                tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right_n)
                tmesh.export(os.path.join(out_mano_path, f'f{fidx}_h{h_idx}.obj'))

            if args.vis:
              for fi, frame in enumerate(frames_buf):
                input_img = frame.astype(np.float32)[:,:,::-1] / 255.0
                if fi in vis_per_frame:
                  vd = vis_per_frame[fi]
                  cam_view = renderer.render_rgba_multiple_osmesa(
                      vd['verts'], cam_t=vd['cam_t'], render_res=vd['img_size'],
                      is_right=vd['right'], mesh_base_color=LIGHT_PURPLE,
                      scene_bg_color=(1, 1, 1), focal_length=scaled_focal_length,
                  )
                  overlay = input_img * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                  writer.append_data((255*overlay).astype(np.uint8)[:, :, ::-1])
                else:
                  writer.append_data((255*input_img).astype(np.uint8)[:, :, ::-1])

          elif args.vis:
            for frame in frames_buf:
              writer.append_data(frame[:,:,::-1])

          frames_buf.clear()
          fidxs_buf.clear()

        print(f"total frames = {total_frames}")
        for fidx in trange(total_frames):
          ret, frame = cap.read()
          if not ret:
            break
          # frames_buf.append(cv.resize(frame, (1280, 720)))
          frames_buf.append(frame)
          fidxs_buf.append(fidx)
          if len(frames_buf) == args.batch_size:
            flush()

        flush()  # process any remaining frames

        cap.release()
        if args.vis: writer.close()
        with open(out_pkl_path, 'wb') as f:
          pickle.dump(all_detections, f)
        print(f"  Saved {len(all_detections)} hand detections → {out_pkl_path}")

if __name__ == '__main__':
  main()
