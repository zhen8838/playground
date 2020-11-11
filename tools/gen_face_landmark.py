import tensorflow as tf
k = tf.keras
import numpy as np
import cv2
from scipy.special import softmax, expit
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())
from tools.auto_anno_movie import get_landmark_annotation, get_base_annotation, LANDMARKS
from tools.manual_anno_movie import save_json
from more_itertools import chunked
import matplotlib.pyplot as plt
import glob


def main(model_path, data_path, landmark_num, batch_size):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  pfld_model: k.Model = k.models.load_model(model_path)
  image_paths = glob.glob(os.path.join(data_path, '*.jpg'))
  for batch_image_path in chunked(image_paths, batch_size):
    batch_image = []
    batch_wh = []
    for image_path in batch_image_path:
      im = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
      batch_wh.append(im.shape[:2][::-1])
      im = cv2.resize(im, (112, 112))
      batch_image.append(im)
    batch_image = tf.stack(batch_image)
    batch_image = ((tf.cast(batch_image, tf.float32) / 255. - 0.5) / 1)
    batch_landmarks = pfld_model.predict(batch_image)
    batch_landmarks = np.reshape(expit(batch_landmarks), (batch_size, -1, 2))
    for landmarks, (w, h), image_path in zip(batch_landmarks, batch_wh, batch_image_path):
      if landmark_num == 5:
        anno = get_base_annotation(image_path, h, w)
        for i, label in zip([96, 97, 54, 76, 82], LANDMARKS):
          # cv2.circle(im, tuple((landmarks[i] * (w, h)).astype('int')), 10, (255, 0, 0))
          x, y = (landmarks[i] * (w, h)).tolist()
          land_anno = get_landmark_annotation(x, y, label, 0)
          anno['shapes'].append(land_anno)
        save_json(anno, image_path.split('.')[0] + '.json')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, help='retina face model path',
                      default='asset/pfld_infer.h5')
  parser.add_argument('--data_path', type=str, help='movie path',
                      default='/home/zqh/workspace/data512x512')
  parser.add_argument('--landmark_num', type=int, choices=[5, 68], default=5)
  parser.add_argument('--batch_size', type=int, default=16)
  args = parser.parse_args()

  main(args.model_path, args.data_path, args.landmark_num, args.batch_size)
