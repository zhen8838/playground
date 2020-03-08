import tensorflow as tf
k = tf.keras
import numpy as np
from typing import List, Tuple
from scipy.special import expit
import cv2


def calc_xy_offset(out_hw: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
  """ for dynamic sacle get xy offset tensor for loss calc

    Parameters
    ----------
    out_hw : tf.Tensor

    Returns
    -------

    [tf.Tensor]

        xy offset : shape [out h , out w , 1 , 2] type=tf.float32

    """
  grid_y = tf.tile(
      tf.reshape(tf.range(0, out_hw[0]), [-1, 1, 1, 1]), [1, out_hw[1], 1, 1])
  grid_x = tf.tile(
      tf.reshape(tf.range(0, out_hw[1]), [1, -1, 1, 1]), [out_hw[0], 1, 1, 1])
  xy_offset = tf.concat([grid_x, grid_y], -1)
  return tf.cast(xy_offset, tf.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray, offset: int = 0,
             method='iou') -> np.ndarray:
  """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    a : np.ndarray

        (n,4) x1,y1,x2,y2

    b : np.ndarray

        (m,4) x1,y1,x2,y2

    offset : int, optional
        by default 0

    method : str, optional
        by default 'iou', can choice ['iou','giou','diou','ciou']

    Returns
    -------
    np.ndarray

        iou (n,m)
    """
  a = a[..., None, :]
  tl = np.maximum(a[..., :2], b[..., :2])
  br = np.minimum(a[..., 2:4], b[..., 2:4])

  area_i = np.prod(np.maximum(br - tl, 0) + offset, axis=-1)
  area_a = np.prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
  area_b = np.prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)

  if method == 'iou':
    return area_i / (area_a+area_b-area_i)
  elif method in ['ciou', 'diou']:
    iou = area_i / (area_a+area_b-area_i)
    outer_tl = np.minimum(a[..., :2], b[..., :2])
    outer_br = np.maximum(a[..., 2:4], b[..., 2:4])
    # two bbox center distance sum((b_cent-a_cent)^2)
    inter_diag = np.sum(
        np.square((b[..., :2] + b[..., 2:]) / 2 - (a[..., :2] + a[..., 2:]) / 2 +
                  offset), -1)
    # two bbox diagonal distance
    outer_diag = np.sum(np.square(outer_tl - outer_br + offset), -1)
    if method == 'diou':
      return iou - inter_diag/outer_diag
    else:
      # calc ciou alpha paramter
      arctan = ((np.math.atan(
          (b[..., 2] - b[..., 0]) / (b[..., 3] - b[..., 1])) - np.math.atan(
              (a[..., 2] - a[..., 0]) / (a[..., 3] - a[..., 1]))))
      v = np.square(2 / np.pi * arctan)
      alpha = v / ((1-iou) + v)
      w_temp = 2 * (a[..., 2] - a[..., 0])
      ar = (8 / np.square(np.pi)) * arctan * ((a[..., 2] - a[..., 0] - w_temp) *
                                              (a[..., 3] - a[..., 1]))
      return np.clip(iou - inter_diag/outer_diag - alpha*ar, -1., 1.)

  elif method in 'giou':
    outer_tl = np.minimum(a[..., :2], b[..., :2])
    outer_br = np.maximum(a[..., 2:4], b[..., 2:4])
    area_o = np.prod(np.maximum(outer_br - outer_tl, 0) + offset, axis=-1)
    union = (area_a + area_b - area_i)
    return (area_i/union) - ((area_o-union) / area_o)


def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float,
                 method='iou') -> np.ndarray:
  """Pure Python NMS oneclass baseline.

    Parameters
    ----------
    bbox : np.ndarray

        bbox, n*(x1,y1,x2,y2)

    score : np.ndarray

        confidence score (n,)

    thresh : float

        nms thresh

    Returns
    -------
    np.ndarray
        keep index
    """
  order = score.argsort()[::-1]
  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    iou = bbox_iou(bbox[i], bbox[order[1:]], method=method)
    inds = np.where(iou <= thresh)[0]
    order = order[inds + 1]

  return keep


def xywh_to_all(grid_pred_xy: tf.Tensor, grid_pred_wh: tf.Tensor,
                out_hw: tf.Tensor, xy_offset: tf.Tensor,
                anchors: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
  """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

    Returns
    -------
    [tf.Tensor, tf.Tensor]

        [all_pred_xy, all_pred_wh]
    """
  all_pred_xy = (tf.sigmoid(grid_pred_xy) + xy_offset) / out_hw[::-1]
  all_pred_wh = tf.exp(grid_pred_wh) * anchors
  return all_pred_xy, all_pred_wh


def validate_ann(clses: tf.Tensor, x1: tf.Tensor, y1: tf.Tensor, x2: tf.Tensor,
                 y2: tf.Tensor, im_w: tf.Tensor, im_h: tf.Tensor) -> tf.Tensor:
  """ when resize or augment img, need validate ann value """
  x1 = tf.clip_by_value(x1, 0, im_w - 1)
  y1 = tf.clip_by_value(y1, 0, im_h - 1)
  x2 = tf.clip_by_value(x2, 0, im_w - 1)
  y2 = tf.clip_by_value(y2, 0, im_h - 1)
  new_ann = tf.concat([clses, x1, y1, x2, y2], -1)
  new_ann.set_shape((None, None))

  bbox_w = new_ann[:, 3] - new_ann[:, 1]
  bbox_h = new_ann[:, 4] - new_ann[:, 2]
  new_ann = tf.boolean_mask(new_ann, tf.logical_and(bbox_w > 1, bbox_h > 1))
  return new_ann


def resize_img(img: tf.Tensor, in_hw: tf.Tensor,
               ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
  """
    resize image and keep ratio

    Parameters
    ----------
    img : tf.Tensor

    ann : tf.Tensor


    Returns
    -------
    [tf.Tensor, tf.Tensor]
        img, ann [ uint8 , float32 ]
    """
  img_hw = tf.shape(img, tf.int64)[:2]
  iw, ih = tf.cast(img_hw[1], tf.float32), tf.cast(img_hw[0], tf.float32)
  w, h = tf.cast(in_hw[1], tf.float32), tf.cast(in_hw[0], tf.float32)
  clses, x1, y1, x2, y2 = tf.split(ann, 5, -1)
  """ transform factor """

  def _resize(img, clses, x1, y1, x2, y2):
    nh = ih * tf.minimum(w / iw, h / ih)
    nw = iw * tf.minimum(w / iw, h / ih)
    dx = (w-nw) / 2
    dy = (h-nh) / 2
    img = tf.image.resize(
        img, [tf.cast(nh, tf.int32), tf.cast(nw, tf.int32)],
        'nearest',
        antialias=True)
    img = tf.image.pad_to_bounding_box(img, tf.cast(dy, tf.int32),
                                       tf.cast(dx, tf.int32),
                                       tf.cast(h, tf.int32), tf.cast(w, tf.int32))
    x1 = x1*nw/iw + dx
    x2 = x2*nw/iw + dx
    y1 = y1*nh/ih + dy
    y2 = y2*nh/ih + dy
    return img, clses, x1, y1, x2, y2

  img, clses, x1, y1, x2, y2 = tf.cond(
      tf.reduce_all(tf.equal(in_hw, img_hw)), lambda:
      (img, clses, x1, y1, x2, y2), lambda: _resize(img, clses, x1, y1, x2, y2))
  new_ann = validate_ann(clses, x1, y1, x2, y2, w, h)
  return img, new_ann


def normlize_img(img: tf.Tensor) -> tf.Tensor:
  """ normlize img """
  return (tf.cast(img, tf.float32) / 255. - 0.5) / 1


def parser_outputs(outputs: List[List[np.ndarray]], orig_hws: List[np.ndarray],
                   obj_thresh: float, nms_thresh: float, iou_method: str,
                   class_num: int, org_out_hw: List[np.ndarray],
                   org_in_hw: List[np.ndarray], xy_offsets: List[tf.Tensor],
                   anchors: List[np.ndarray]
                  ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """ yolo parser one image output

        outputs : batch * [box,clss,score]

        box : [x1, y1, x2, y2]
        clss : [class]
        score : [score]
    """
  results = []
  anchor_number = len(anchors[0])
  for y_pred, orig_hw in zip(outputs, orig_hws):
    # In order to ensure the consistency of the framework code reshape here.
    y_pred = [
        np.reshape(pred,
                   list(pred.shape[:-1]) + [anchor_number, 5 + class_num])
        for pred in y_pred
    ]
    """ box list """
    _xyxy_box = []
    _xyxy_box_scores = []
    """ preprocess label """
    for l, pred_label in enumerate(y_pred):
      """ split the label """
      pred_xy = pred_label[..., 0:2]
      pred_wh = pred_label[..., 2:4]
      pred_confidence = pred_label[..., 4:5]
      pred_cls = pred_label[..., 5:]
      if class_num > 1:
        box_scores = expit(pred_cls) * expit(pred_confidence)
      else:
        box_scores = expit(pred_confidence)
      """ reshape box  """
      # NOTE tf_xywh_to_all will auto use sigmoid function
      pred_xy_A, pred_wh_A = xywh_to_all(pred_xy, pred_wh, org_out_hw[l],
                                         xy_offsets[l], anchors[l])
      # NOTE boxes from xywh to xyxy
      boxes = np.concatenate((pred_xy_A.numpy(), pred_wh_A.numpy()), -1)
      boxes = boxes * np.tile(org_in_hw[::-1], [2])
      boxes[..., :2] -= boxes[..., 2:] / 2
      boxes[..., 2:] += boxes[..., :2]
      # NOTE reverse boxes to orginal image scale
      scale = np.min(org_in_hw / orig_hw)
      xy_off = ((org_in_hw - orig_hw*scale) / 2)[::-1]
      boxes = (boxes - np.tile(xy_off, [2])) / scale
      boxes = np.reshape(boxes, (-1, 4))
      box_scores = np.reshape(box_scores, (-1, class_num))
      """ append box and scores to global list """
      _xyxy_box.append(boxes)
      _xyxy_box_scores.append(box_scores)

    xyxy_box = np.concatenate(_xyxy_box, axis=0)
    xyxy_box_scores = np.concatenate(_xyxy_box_scores, axis=0)

    mask = xyxy_box_scores >= obj_thresh
    """ do nms for every classes"""
    _boxes = []
    _scores = []
    _classes = []
    for c in range(class_num):
      class_boxes = xyxy_box[mask[:, c]]
      class_box_scores = xyxy_box_scores[:, c][mask[:, c]]
      select = nms_oneclass(
          class_boxes, class_box_scores, nms_thresh, method=iou_method)
      class_boxes = class_boxes[select]
      class_box_scores = class_box_scores[select]
      _boxes.append(class_boxes)
      _scores.append(class_box_scores)
      _classes.append(np.ones_like(class_box_scores) * c)

    box: np.ndarray = np.concatenate(_boxes, axis=0)
    clss: np.ndarray = np.concatenate(_classes, axis=0)
    score: np.ndarray = np.concatenate(_scores, axis=0)
    results.append([box, clss, score])
  return results


colormap = [
    (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133),
    (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), (0, 143, 255),
    (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
    (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245),
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
    (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224),
    (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214),
    (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
    (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7), (255, 122, 8),
    (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255),
    (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15), (20, 255, 0),
    (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
    (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200)
]


def draw_image(img: np.ndarray, ann: np.ndarray, is_show=True,
               scores=None) -> np.ndarray:
  """ draw img and show bbox , set ann = None will not show bbox

    Parameters
    ----------
    img : np.ndarray

    ann : np.ndarray

        scale is all image pixal scale
        shape : [p,x1,y1,x2,y2]

    is_show : bool

        show image
    """
  p = ann[:, 0]
  xyxybox = ann[:, 1:]
  for i, a in enumerate(xyxybox):
    classes = int(p[i])
    r_top = tuple(a[0:2].astype(int))
    l_bottom = tuple(a[2:].astype(int))
    r_bottom = (r_top[0], l_bottom[1])
    org = (np.maximum(np.minimum(r_bottom[0], img.shape[1] - 12), 0),
           np.maximum(np.minimum(r_bottom[1], img.shape[0] - 12), 0))
    cv2.rectangle(img, r_top, l_bottom, colormap[classes])
    if isinstance(scores, np.ndarray):
      cv2.putText(
          img,
          f'{classes} {scores[i]:.2f}',
          org,
          cv2.FONT_HERSHEY_SIMPLEX,
          0.75,
          colormap[classes],
          thickness=1)
    else:
      cv2.putText(
          img,
          f'{classes}',
          org,
          cv2.FONT_HERSHEY_SIMPLEX,
          0.75,
          colormap[classes],
          thickness=1)

  return img


if __name__ == "__main__":
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  draw_imgs = []
  org_in_hw = np.array([416, 416])
  org_out_hw = np.array([[13, 13], [26, 26], [52, 52]])
  obj_thresh = 0.45
  nms_thresh = 0.3
  iou_method = 'diou'
  class_num = 20
  xy_offsets: List[tf.Tensor] = [
      calc_xy_offset(org_out_hw[i]) for i in range(len(org_out_hw))
  ]
  anchors = np.load('yolov3_camera/voc_anchor_v3.npy', allow_pickle=True)
  infer_model: k.Model = k.models.load_model('yolov3_camera/infer_model_203.h5')

  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  orig_hws = np.array([(480, 640)])
  while (True):
    ret, draw_img = capture.read()
    det_img, _ = resize_img(
        cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), org_in_hw,
        np.zeros([0, 5], 'float32'))
    det_img = normlize_img(det_img)
    det_imgs = det_img[None, ...]
    outputs = infer_model.predict(det_imgs, len(orig_hws))
    # NOTE change outputs List to n*[layer_num*[arr]]
    outputs = [[output[i] for output in outputs] for i in range(len(orig_hws))]
    results = parser_outputs(outputs, orig_hws, obj_thresh, nms_thresh,
                             iou_method, class_num, org_out_hw, org_in_hw,
                             xy_offsets, anchors)
    bbox, cals, scores = results[0]
    draw_img = draw_image(
        draw_img, np.hstack([cals[:, None], bbox]), scores=scores)
    cv2.imshow('frame', draw_img)
    if cv2.waitKey(1) == ord('q'):
      break
