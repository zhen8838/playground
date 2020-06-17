import tensorflow as tf
k = tf.keras
import sys
sys.path.insert(0, '.')
from retinaface_camera import get_anchors, nms_oneclass, decode_bbox, decode_landm
from scipy.special import softmax
import numpy as np
import cv2
from typing import List
import argparse


def letter_box_resize(img, in_hw):
  img_hw = np.array(img.shape[:2], 'int32')
  scale = np.min(in_hw / img_hw)

  # NOTE calc the x,y offset
  yx_off = ((in_hw - img_hw * scale) / 2).astype('int32')

  img_hw = (img_hw * scale).astype('int32')

  img = cv2.resize(img, tuple(img_hw[::-1]))
  img = np.pad(img, [[yx_off[0], in_hw[0] - img_hw[0] - yx_off[0]],
                     [yx_off[1], in_hw[1] - img_hw[1] - yx_off[1]], [0, 0]])
  return img


def reverse_letter_box(bbox: np.ndarray, landm: np.ndarray, in_hw: np.ndarray,
                       img_hw: np.ndarray) -> np.ndarray:
  """rescae predict box to orginal image scale

    """
  scale = np.min(in_hw / img_hw)
  xy_off = ((in_hw - img_hw * scale) / 2)[::-1]

  bbox = (bbox - np.tile(xy_off, [2])) / scale
  landm = (landm - np.tile(xy_off, [5])) / scale
  return bbox, landm


class RetinaFace():

  def __init__(self,
               model_path: str,
               in_hw: tuple,
               obj_thresh=0.8,
               nms_threshold=0.4,
               variances=[0.1, 0.2]):
    self.model: k.Model = k.models.load_model(model_path)
    self.in_hw = in_hw
    self.anchors: np.ndarray = get_anchors(
        in_hw, [[0.025, 0.05], [0.1, 0.2], [0.4, 0.8]], [8, 16, 32])
    self.obj_thresh = obj_thresh
    self.nms_threshold = nms_threshold
    self.variances = variances

  def detect_faces(self, draw_img):
    """ resize """
    img = np.copy(draw_img)
    orig_hw = np.array(img.shape[:2], 'int32')
    img = letter_box_resize(img, self.in_hw)
    """ normlize """
    img = (img / 255. - 0.5) / 1
    """ infer """
    predictions = self.model.predict(img[None, ...])
    """ parser """
    bbox, landm, clses = np.split(predictions[0], [4, -2], 1)
    """ softmax class"""
    clses = softmax(clses, -1)
    score = clses[:, 1]
    """ decode """
    bbox = decode_bbox(bbox, self.anchors, self.variances)
    bbox = bbox * np.repeat([640, 640], 2)
    """ landmark """
    landm = decode_landm(landm, self.anchors, self.variances)
    landm = landm * np.repeat([640, 640], 5)
    """ filter low score """
    inds = np.where(score > self.obj_thresh)[0]
    bbox = bbox[inds]
    landm = landm[inds]
    score = score[inds]
    """ keep top-k before NMS """
    order = np.argsort(score)[::-1]
    bbox = bbox[order]
    landm = landm[order]
    score = score[order]
    """ do nms """
    keep = nms_oneclass(bbox, score, self.nms_threshold)

    bbox = bbox[keep]
    landm = landm[keep]
    score = score[keep]

    bbox, landm = reverse_letter_box(bbox, landm, self.in_hw, orig_hw)

    return bbox, landm, score

  def detect_faces_and_crop(self, draw_img, crop_ratio=1.):
    orig_wh = draw_img.shape[1:: -1]
    bboxs, landms, scores = self.detect_faces(draw_img)
    face_imgs = []
    face_landmarks = []
    vaild_bboxs = []
    for box, landm, score in zip(bboxs.astype(int), landms, scores):
      # crop face region
      cx, cy = (box[:2] + box[2:]) // 2
      halfw = int((np.max(box[2:] - box[:2]) // 2) * crop_ratio)
      face_img: np.ndarray = draw_img[np.maximum(cy - halfw, 0):
                                      np.minimum(cy + halfw, orig_wh[1]),
                                      np.maximum(cx - halfw, 0):
                                      np.minimum(cx + halfw, orig_wh[0])]
      face_img_wh = face_img.shape[1::-1]
      if face_img_wh[0] != face_img_wh[1]:
        top = np.maximum(-(cy - halfw), 0)
        bottom = np.maximum(cy + halfw - orig_wh[1], 0)
        left = np.maximum(-(cx - halfw), 0)
        right = np.maximum(cx + halfw - orig_wh[0], 0)
        face_img = cv2.copyMakeBorder(face_img, top, bottom, left,
                                      right, cv2.BORDER_CONSTANT, value=0)
      if min(face_img_wh) > 10:
        face_landm = np.reshape(landm, (-1, 2)) - np.array(
            [cx - halfw, cy - halfw], 'int32')
        face_imgs.append(face_img)
        face_landmarks.append(face_landm)
        vaild_bboxs.append(box)
    return vaild_bboxs, face_imgs, face_landmarks

  @staticmethod
  def face_algin_by_landmark(face_img: np.ndarray, face_landmark: np.ndarray,
                             TEMPLATE: np.ndarray) -> np.ndarray:
    img_dim = face_img.shape[:2][::-1]
    M, _ = cv2.estimateAffinePartial2D(face_landmark, img_dim * TEMPLATE)
    warped_img = cv2.warpAffine(face_img, M, img_dim)
    return warped_img

  @staticmethod
  def draw_boxes_with_title(draw_img, bboxs, titles):
    for i, bbox in enumerate(bboxs):
      lt = tuple(bbox[:2].astype(int))
      rb = tuple(bbox[2:].astype(int))
      cv2.rectangle(draw_img, lt, rb, (255, 0, 0), 2)
      cv2.putText(
          draw_img,
          titles[i], (lt[0], rb[1]),
          cv2.FONT_HERSHEY_SIMPLEX,
          1, (0, 255, 0),
          thickness=1)
    return draw_img


class FaceRec(object):

  def __init__(self,
               model_path: str,
               in_hw: tuple,
               threshold: float,
               database_path: str = 'asset/test_face_data/example.npy'):
    self.in_hw = in_hw
    self.model: tf.lite.Interpreter = tf.lite.Interpreter(model_path)
    self.model.allocate_tensors()
    self.input_details = self.model.get_input_details()[0]['index']
    self.output_details = self.model.get_output_details()[0]['index']
    self.threshold = threshold
    if database_path:
      emmbeds_labels = np.load(database_path, allow_pickle=True)[()]
      self.emmbeds: np.ndarray = emmbeds_labels['emmbeds']
      self.labels: np.ndarray = emmbeds_labels['labels']

  def forward(self, img):
    img = np.copy(img)
    img = cv2.resize(img, tuple(self.in_hw))
    img = np.expand_dims(img, 0)
    img = (img / 255. - 0.5 / 0.5019607843137255).astype('float32')
    self.model.set_tensor(self.input_details, img)
    self.model.invoke()
    emmbeding = self.model.get_tensor(self.output_details)
    return emmbeding

  @staticmethod
  def l2_distance(pred_emmbeds, emmbeds):
    diff = np.sum(np.square(pred_emmbeds[:, None, :] - emmbeds), -1)
    return diff

  @staticmethod
  def normlize(arr):
    return arr / np.linalg.norm(arr, 2, -1, keepdims=True)

  def get_pred_labels(self, pred_emmbeds):
    diff = self.l2_distance(pred_emmbeds, self.emmbeds)
    pred_idx = np.argmin(diff, -1)
    pred_score = np.min(diff, -1)
    masks = pred_score > self.threshold
    pred_labels = self.labels[pred_idx]
    pred_labels[masks] = 'None'
    return pred_labels


def main(label_path='asset/test_face_data/index.csv',
         save_path='asset/test_face_data/example.npy'):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  img_paths_labels = np.loadtxt(label_path, dtype=str, delimiter=' ')
  img_paths = img_paths_labels[:, 0]
  img_labels = img_paths_labels[:, 1]
  """ init retinaface """
  retinaface = RetinaFace('asset/retinaface_train.h5', [640, 640])
  TEMPLATE = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393],
                       [0.500225, 0.64050536], [0.37097589, 0.82469196],
                       [0.631517, 0.82325089]])
  """ init facerec """
  mbfacenet = FaceRec('asset/mbv1face.tflite', [112, 112], 1.45, None)

  valid_emmbeds = []
  valid_labels = []
  for i, img_path in enumerate(img_paths):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    bbox, face_imgs, face_landmarks = retinaface.detect_faces_and_crop(img)
    if len(bbox) != 1:
      print(f'Image {img_path} must only have one face! Please change this photo')
      continue
    else:
      warped_img = retinaface.face_algin_by_landmark(face_imgs[0],
                                                     face_landmarks[0], TEMPLATE)
      emmbed = mbfacenet.forward(warped_img)
      valid_emmbeds.append(emmbed)
      valid_labels.append(img_labels[i])

  valid_emmbeds = mbfacenet.normlize(np.concatenate(valid_emmbeds, 0))
  valid_labels = np.array(valid_labels)

  np.save(
      save_path, {
          'emmbeds': valid_emmbeds,
          'labels': valid_labels
      },
      allow_pickle=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--label_path',
      type=str,
      help='label file path',
      default='asset/test_face_data/index.csv')
  parser.add_argument(
      '--save_path',
      type=str,
      help='save database file path',
      default='asset/test_face_data/example.npy')
  args = parser.parse_args()

  main(args.label_path, args.save_path)
