import os
import sys
sys.path.insert(0, os.getcwd())
from face_seg import FaceSeg
from facerec_camera.make_database import RetinaFace
import cv2
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse


def main(data_path, save_path, use_face_crop, use_face_mask, face_crop_ratio):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  data_path = Path(data_path)
  save_path = Path(save_path)
  if not save_path.exists():
    save_path.mkdir(parents=True)

  """ init retinaface """
  retinaface = RetinaFace('asset/retinaface_train.h5', [640, 640], crop_ratio=face_crop_ratio)
  TEMPLATE = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393],
                       [0.500225, 0.64050536], [0.37097589, 0.82469196],
                       [0.631517, 0.82325089]])
  """ init face seg """
  faceseg = FaceSeg('asset/seg_model_saved', [384, 384])

  generator = torch.jit.load('asset/genA2B.pth')
  generator.eval()

  for data_img_path in data_path.iterdir():
    orig_img = cv2.cvtColor(cv2.imread(data_img_path.as_posix(),
                                       cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if use_face_crop:
      bbox, face_imgs, face_landmarks = retinaface.detect_one_face_and_crop(
          orig_img)
      if len(bbox) < 1:
        continue
      warped_img = face_imgs[0]
    else:
      warped_img = orig_img

    mask_img = (warped_img if warped_img.shape[:2] == [
                256, 256] else cv2.resize(warped_img, (256, 256),
                                          interpolation=cv2.INTER_AREA))

    mask_imgs = mask_img[None, ...]
    if use_face_mask:
      mask = faceseg.get_mask(mask_imgs)[0]
      mask = mask / 255.
      face_white_bg = (mask_img * mask + (1 - mask) * 255).astype(np.uint8)
    else:
      face_white_bg = mask_img

    face = np.transpose(face_white_bg[None, ...], (0, 3, 1, 2))
    face = (face / 127.5 - 1).astype(np.float32)
    face = torch.from_numpy(face).to('cpu')
    with torch.no_grad():
      cartoon = generator(face)[0][0]
    cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
    cartoon = (cartoon + 1) * 127.5
    if use_face_mask:
      cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
    else:
      cartoon = cartoon.astype(np.uint8)
    save_img_path = save_path / (data_img_path.stem + '.png')
    save_img = np.concatenate([cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR),
                               cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)], 1)
    cv2.imwrite(save_img_path.as_posix(), save_img)
    print(data_img_path, '-->', save_img_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, help='photo folder path',
                      default='asset/test_phototransfer')
  parser.add_argument('--save_path', type=str, help='save folder path',
                      default='test/test_phototransfer')
  parser.add_argument('--use_face_mask', type=str, help='weather use face crop',
                      choices=['True', 'False'], default='False')

  parser.add_argument('--use_face_crop', type=str, help='weather use face crop',
                      choices=['True', 'False'], default='False')

  parser.add_argument('--face_crop_ratio', type=float, help='face crop ratio', default=1.25)
  args = parser.parse_args()

  main(args.data_path, args.save_path,
       eval(args.use_face_crop),
       eval(args.use_face_mask),
       args.face_crop_ratio)
