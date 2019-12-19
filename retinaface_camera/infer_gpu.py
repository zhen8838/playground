import tensorflow as tf
k = tf.keras
import numpy as np
import cv2
from scipy.special import softmax


def get_anchors(in_hw,
                anchor_widths,
                anchor_steps):
    """ get anchors """
    feature_maps = [[round(in_hw[0] / step), round(in_hw[1] / step)] for step in anchor_steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        anchor_width = anchor_widths[k]
        feature = np.empty((f[0], f[1], len(anchor_width), 4))
        for i in range(f[0]):
            for j in range(f[1]):
                for n, width in enumerate(anchor_width):
                    s_kx = width
                    s_ky = width
                    cx = (j + 0.5) * anchor_steps[k] / in_hw[1]
                    cy = (i + 0.5) * anchor_steps[k] / in_hw[0]
                    feature[i, j, n, :] = cx, cy, s_kx, s_ky
        anchors.append(feature)

    anchors = np.concatenate([
        np.reshape(anchors[0], (-1, 4)),
        np.reshape(anchors[1], (-1, 4)),
        np.reshape(anchors[2], (-1, 4))], 0)

    return np.clip(anchors, 0, 1).astype('float32')


def decode_bbox(bbox, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """

    boxes = np.concatenate((
        anchors[:, :2] + bbox[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(landm, anchors, variances):
    """Decode landm from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """
    landms = np.concatenate(
        (anchors[:, :2] + landm[:, :2] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 2:4] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 4:6] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 6:8] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 8:10] * variances[0] * anchors[:, 2:]), 1)
    return landms


def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float) -> np.ndarray:
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
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def detect_face(model: k.Model,
                anchors: np.ndarray,
                draw_img: np.ndarray,
                obj_thresh=0.7,
                nms_threshold=0.4,
                variances=[0.1, 0.2]):
    """ resize """
    img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    """ normlize """
    img = (img / 255. - 0.5) / 1

    """ infer """
    predictions = model.predict(img[None, ...])
    """ parser """
    bbox, landm, clses = np.split(predictions[0], [4, -2], 1)
    """ softmax class"""
    clses = softmax(clses, -1)
    score = clses[:, 1]
    """ decode """
    bbox = decode_bbox(bbox, anchors, variances)
    bbox = bbox * np.repeat([640, 640], 2)
    """ landmark """
    landm = decode_landm(landm, anchors, variances)
    landm = landm * np.repeat([640, 640], 5)
    """ filter low score """
    inds = np.where(score > obj_thresh)[0]
    bbox = bbox[inds]
    landm = landm[inds]
    score = score[inds]
    """ keep top-k before NMS """
    order = np.argsort(score)[::-1]
    bbox = bbox[order]
    landm = landm[order]
    score = score[order]
    """ do nms """
    keep = nms_oneclass(bbox, score, nms_threshold)

    bbox = bbox[keep]
    landm = landm[keep]
    score = score[keep]

    for i, flag in enumerate(np.ones_like(score[:, None])):
        if flag == 1:
            cv2.rectangle(draw_img, tuple(bbox[i][:2].astype(int)),
                          tuple(bbox[i][2:].astype(int)), (255, 0, 0), 2)
            for ldx, ldy, color in zip(landm[i][0::2].astype(int),
                                       landm[i][1::2].astype(int),
                                       [(255, 0, 0), (0, 255, 0),
                                        (0, 0, 255), (255, 255, 0),
                                        (255, 0, 255)]):
                cv2.circle(draw_img, (ldx, ldy), 3, color, 2)
    return draw_img


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model: k.Model = k.models.load_model('asset/retinaface_train.h5')
    anchors = get_anchors([640, 640],
                          [[0.025, 0.05], [0.1, 0.2], [0.4, 0.8]],
                          [8, 16, 32])
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while(True):
        ret, img = capture.read()
        # img = cv2.flip(img, 1)
        img = cv2.copyMakeBorder(img, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=0)
        draw_img = detect_face(model, anchors, img)
        cv2.imshow('frame', draw_img)
        if cv2.waitKey(1) == ord('q'):
            break
