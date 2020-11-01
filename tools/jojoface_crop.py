import os
import sys
sys.path.insert(0, os.getcwd())
from facerec_camera.make_database import RetinaFace
from tools.auto_anno_movie import get_base_annotation, get_landmark_annotation, LANDMARKS
from tools.manual_anno_movie import save_json
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
import cv2


def main(model_path, movie_path, save_path_root, movie_id, batch_size: int,
         face_crop_ratio: float, skip_frame: int, ignore_size: int, brightness_thresh: int):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  movie_path = Path(movie_path)
  save_path_root = Path(save_path_root)

  """ init retinaface """
  retinaface = RetinaFace(model_path, [640, 640],
                          crop_ratio=face_crop_ratio,
                          fill_value=255,
                          ignore_size=ignore_size)

  save_path = movie_path.parent.name + '_' + f'{movie_id}' + '_'
  if not save_path_root.exists():
    save_path_root.mkdir(parents=True)
  m = 0
  stream = cv2.VideoCapture(movie_path.as_posix())
  length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
  print(movie_path, 'Frames = ', length)
  while True:
    ret = True
    frames = []
    for _ in range(batch_size):
      ret, frame = stream.read()
      frame_id = int(stream.get(cv2.CAP_PROP_POS_FRAMES))
      stream.set(cv2.CAP_PROP_POS_FRAMES, frame_id + skip_frame)
      frames.append(frame)
    if not ret:
      break
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    vaild_bboxss, face_imgss, face_landmarkss = retinaface.detect_faces_and_crop(frames)
    for vaild_bboxs, face_imgs, face_landmarks in zip(vaild_bboxss, face_imgss, face_landmarkss):
      n = 0
      for vaild_bbox, face_img, face_landmark in zip(vaild_bboxs, face_imgs, face_landmarks):
        w, h = face_img.shape[1::-1]  # to [0,1]
        landmark_scale = face_landmark / [w, h]
        # face_img = cv2.resize(face_img, (256, 256))
        im_path = (save_path_root / (save_path + f'{m:d}_{n:d}.jpg')).as_posix()
        js_path = (save_path_root / (save_path + f'{m:d}_{n:d}.json')).as_posix()
        anno = get_base_annotation(im_path, h, w)
        # ensure face always is front
        if (abs(landmark_scale[0][0] - landmark_scale[1][0]) < 0.14 or
                abs(landmark_scale[0][1] - landmark_scale[1][1]) > 0.03):
          continue
        # ensure brightness
        if face_img.mean() < brightness_thresh:
          continue
          # print(im_path, "mean brightness is ", face_img.mean())
          # brightness = min((255 - face_img.max()) - 20, 150)
          # if brightness > 0:
          #   face_img += brightness
          # else:
          #   continue
        for idx, label in enumerate(LANDMARKS):
          x, y = face_landmark[idx].tolist()
          anno['shapes'].append(get_landmark_annotation(x, y, label, 0))
        cv2.imwrite(im_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        save_json(anno, js_path)
        n += 1
    m += 1


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, help='retina face model path',
                      default='asset/retinaface_train.h5')
  parser.add_argument('--data_path', type=str, help='movie path',
                      default='/media/zqh/Documents/JOJO4/[AGE-JOJO&UHA-WING&Kamigami][160073][01][BD-720P][CHS-JAP] AVC.mp4')
  parser.add_argument('--save_path', type=str, help='save folder path',
                      default='/media/zqh/Documents/JOJO_face_crop')
  parser.add_argument('--movie_id', type=str, help='save movie id', default='01')
  parser.add_argument('--batch_size', type=int, help='batch size', default=24)
  parser.add_argument('--face_crop_ratio', type=float, help='face crop ratio', default=1.5)
  parser.add_argument('--skip_frame', type=int, help='skip frame num', default=5)
  parser.add_argument('--ignore_size', type=int, help='minimum face size', default=224)
  parser.add_argument('--brightness_thresh', type=int, help='minimum brightness', default=120)
  args = parser.parse_args()

  main(args.model_path, args.data_path, args.save_path, args.movie_id,
       args.batch_size, args.face_crop_ratio, args.skip_frame,
       args.ignore_size, args.brightness_thresh)
