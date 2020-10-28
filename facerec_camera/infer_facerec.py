import tensorflow as tf
k = tf.keras
import sys
sys.path.insert(0, '.')
from facerec_camera.make_database import RetinaFace, FaceRec
from scipy.special import softmax
import numpy as np
import cv2
from typing import List
import argparse


def main():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  """ init retinaface """
  retinaface = RetinaFace('asset/retinaface_train.h5', [640, 640])
  TEMPLATE = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393],
                       [0.500225, 0.64050536], [0.37097589, 0.82469196],
                       [0.631517, 0.82325089]])
  """ init facerec """
  mbfacenet = FaceRec('asset/mbv1face.tflite', [112, 112], 1.45)

  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  while (True):
    ret, draw_img = capture.read()
    img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    bbox, face_imgs, face_landmarks = retinaface.detect_one_face_and_crop(img)
    if len(bbox) > 0:
      warped_imgs = [
          retinaface.face_algin_by_landmark(face_img, face_landmark, TEMPLATE)
          for face_img, face_landmark in zip(face_imgs, face_landmarks)
      ]

      pred_emmbeds = mbfacenet.normlize(
          np.concatenate(
              [mbfacenet.forward(warped_img) for warped_img in warped_imgs], 0))

      pred_labels = mbfacenet.get_pred_labels(pred_emmbeds)

      draw_img = retinaface.draw_boxes_with_title(draw_img, bbox, pred_labels)

    cv2.imshow('frame', draw_img)
    if cv2.waitKey(1) == ord('q'):
      break


if __name__ == "__main__":
  main()
