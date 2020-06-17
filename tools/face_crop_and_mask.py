import os
import sys
sys.path.insert(0, os.getcwd())
from face_seg import FaceSeg
from facerec_camera.make_database import RetinaFace
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
import cv2
from tqdm import trange


def main(data_path, save_path, batch_size: int,
         use_face_crop: bool,
         face_crop_ratio: float):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  data_path = Path(data_path)
  save_path = Path(save_path)

  if not save_path.exists():
    save_path.mkdir(parents=True)

  """ init retinaface """
  retinaface = RetinaFace('asset/retinaface_train.h5', [640, 640])
  TEMPLATE = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393],
                       [0.500225, 0.64050536], [0.37097589, 0.82469196],
                       [0.631517, 0.82325089]])
  """ init face seg """
  faceseg = FaceSeg('asset/seg_model_saved', [384, 384])
  data_img_paths = [p for p in data_path.iterdir()]
  save_img_paths = [(save_path / p.name).with_suffix('.png') for p in data_img_paths]

  for i in trange(0, len(data_img_paths), batch_size):
    mask_imgs = []
    valid_save_path = []
    for data_path, save_path in zip(data_img_paths[i:i + batch_size],
                                    save_img_paths[i:i + batch_size]):
      orig_img = cv2.cvtColor(cv2.imread(data_path.as_posix(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
      if use_face_crop:
        bbox, face_imgs, face_landmarks = retinaface.detect_faces_and_crop(
            orig_img, face_crop_ratio)
        if len(bbox) < 1:
          continue
        warped_img = retinaface.face_algin_by_landmark(face_imgs[0], face_landmarks[0], TEMPLATE)
        mask_imgs.append(warped_img if warped_img.shape[:2] == [
                         512, 512] else cv2.resize(warped_img, (512, 512)))
      else:
        mask_imgs.append(orig_img if orig_img.shape[:2] == [
                         512, 512] else cv2.resize(orig_img, (512, 512)))
      valid_save_path.append(save_path)

    if len(mask_imgs) > 0:
      mask_imgs = np.stack(mask_imgs)
      masks = faceseg.get_mask(mask_imgs)
      for mask_img, mask, path in zip(mask_imgs, masks, valid_save_path):
        mask = mask / 255.
        face_white_bg = (mask_img * mask + (1 - mask) * 255).astype(np.uint8)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, help='photo folder path',
                      default='/home/zqh/workspace/data512x512')
  parser.add_argument('--save_path', type=str, help='save folder path',
                      default='/home/zqh/workspace/data512x512_masked')
  parser.add_argument('--batch_size', type=int, help='batch size', default=4)
  parser.add_argument('--use_face_crop', type=str, help='weather use face crop',
                      choices=['True', 'False'], default='False')
  parser.add_argument('--face_crop_ratio', type=float, help='face crop ratio', default=2.)
  args = parser.parse_args()

  main(args.data_path, args.save_path, args.batch_size,
       eval(args.use_face_crop), args.face_crop_ratio)
