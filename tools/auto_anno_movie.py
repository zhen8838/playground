import os
import sys
sys.path.insert(0, os.getcwd())
from facerec_camera.make_database import RetinaFace
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, AnyStr, Any, Tuple
import cv2
import json


class AnnoationSub(object):
  label: str
  points: List[Tuple[float, float]]
  group_id: int
  shape_type: str
  flags: Dict


class AnnoationBase(object):
  version: str
  flags: Dict
  shapes: List[AnnoationSub]
  imagePath: str
  imageData: str
  imageHeight: str
  imageWidth: str


def get_face_annotation(xmin: float, ymin: float,
                        xmax: float, ymax: float,
                        group_id: int) -> AnnoationSub:
  return {'label': 'face',
          'points': [[xmin.item(), ymin.item()],
                     [xmax.item(), ymax.item()]],
          'group_id': group_id,
          'shape_type': 'rectangle',
          'flags': {}}


def get_point_annotation(x: float, y: float,
                         label: str,
                         group_id: int) -> AnnoationSub:
  return {'label': label,
          'points': [[x, y]],
          'group_id': group_id,
          'shape_type': 'point',
          'flags': {}}


def get_base_annotation(path: str, Height: int, Width: int) -> AnnoationBase:
  return {'version': '4.5.6',
          'flags': {},
          'shapes': [],
          'imagePath': path,
          'imageData': None,
          'imageHeight': Height,
          'imageWidth': Width}


LANDMARKS = ['left_eye', 'right_eye',
             'nose', 'left_mouth',
             'right_mouth']


def main(output_dir, movie_path: str, movie_id: str):
  # output_dir = '/media/zqh/Documents/jojo-face-landmark'
  # input_dir = '/media/zqh/Documents/JOJO4'
  # predictor = FaceBoxesPredict('FaceBoxes_epoch_90.pth', confidenceTh=0.7)
  predictor = RetinaFace('asset/retinaface_train.h5', [640, 640])
  stream = cv2.VideoCapture(movie_path)
  n = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
  output_path = os.path.join(output_dir, movie_id)
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  print(movie_path, 'Frames = ', n)
  while True:
    ret, frame = stream.read()
    frame_id = int(stream.get(cv2.CAP_PROP_POS_FRAMES))
    if not ret:
      break
    """ do some thing """
    bboxs, landms, score = predictor.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if len(bboxs) > 0:
      frame_name = f'{frame_id:d}'
      image_name = frame_name + '.jpg'
      json_name = frame_name + '.json'
      anno = get_base_annotation(image_name, *frame.shape[:2])
      for group_id, (bbox, landm) in enumerate(zip(bboxs, landms)):
        face = get_face_annotation(*bbox, group_id)
        anno['shapes'].append(face)
        # NOTE 导出landmark
        xymin = np.array(face['points'][0])
        xymax = np.array(face['points'][1])
        wh = xymax - xymin
        for point, label in zip(np.reshape(landm, (-1, 2)), LANDMARKS):
          x, y = point
          point_anno = get_point_annotation(x.item(), y.item(), label, group_id)
          anno['shapes'].append(point_anno)

      text = json.dumps(anno, indent=4)
      with open(os.path.join(output_path, json_name), 'w') as f:
        f.write(text)
      cv2.imwrite(os.path.join(output_path, image_name), frame)
      """ skip frame """
    stream.set(cv2.CAP_PROP_POS_FRAMES, frame_id + 30)
  stream.release()


if __name__ == "__main__":
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

  parser = argparse.ArgumentParser(description='')

  parser.add_argument('--output_dir', required=True,
                      default='/media/zqh/Documents/jojo-face-landmark', type=str)
  parser.add_argument('--movie_path', required=True, default='/media/zqh/Documents/JOJO4', type=str)
  parser.add_argument('--movie_id', required=True, default='00', type=str)

  args = parser.parse_args()
  main(args.output_dir, args.movie_path, args.movie_id)
  """ 
  python tools/auto_anno_movie.py  --output_dir="/media/zqh/Documents/jojo-face-landmark" \
  --movie_path="/media/zqh/Documents/JOJO4/[AGE-JOJO&UHA-WING&Kamigami][160073][03][BD-720P][CHS-JAP] AVC.mp4" \
  --movie_id="03"
  """
