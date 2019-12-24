import tensorflow as tf
k = tf.keras
import numpy as np
import cv2
from scipy.special import softmax, expit
from retinaface_camera import get_anchors, nms_oneclass, decode_bbox, decode_landm


def detect_face(retinaface_model: k.Model,
                pfld_model: k.Model,
                anchors: np.ndarray,
                draw_img: np.ndarray,
                obj_thresh=0.7,
                nms_threshold=0.4,
                variances=[0.1, 0.2]):
    """ resize """
    img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    """ normlize """
    det_img = ((img / 255. - 0.5) / 1)[None, ...]
    """ infer """
    predictions = retinaface_model.predict(det_img)
    """ parser """
    bbox, landm, clses = np.split(predictions[0], [4, -2], 1)
    """ softmax class"""
    clses = softmax(clses, -1)
    score = clses[:, 1]
    """ decode """
    bbox = decode_bbox(bbox, anchors, variances)
    bbox = bbox * np.tile([640, 640], [2])
    """ filter low score """
    inds = np.where(score > obj_thresh)[0]
    bbox = bbox[inds]
    score = score[inds]
    """ keep top-k before NMS """
    order = np.argsort(score)[::-1]
    bbox = bbox[order]
    score = score[order]
    """ do nms """
    keep = nms_oneclass(bbox, score, nms_threshold)

    for b, s in zip(bbox[keep].astype(int), score[keep]):
        cv2.rectangle(draw_img, tuple(b[:2]), tuple(b[2:]),
                      (255, 0, 0), 2)
        cx, cy = (b[:2] + b[2:]) // 2
        halfw = np.max(b[2:] - b[:2]) // 2
        croped_img: np.ndarray = img[cy - halfw:cy + halfw, cx - halfw:cx + halfw]
        croped_wh = croped_img.shape[1::-1]
        if croped_wh[0] == croped_wh[1] and min(croped_wh) > 10:
            croped_img = cv2.resize(croped_img, (112, 112))
            croped_img = ((croped_img / 255. - 0.5) / 1)[None, ...]
            landmarks = pfld_model.predict(croped_img)
            s_point = np.array([cx - halfw, cy - halfw])
            for landm in np.reshape(expit(landmarks), (-1, 2)) * croped_wh:
                cv2.circle(draw_img, tuple((s_point + landm).astype(int)), 1, (0, 255, 0))

    return draw_img


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    retinaface_model: k.Model = k.models.load_model('asset/retinaface_train.h5')
    anchors = get_anchors([640, 640],
                          [[0.025, 0.05], [0.1, 0.2], [0.4, 0.8]],
                          [8, 16, 32])
    pfld_model: k.Model = k.models.load_model('asset/pfld_infer.h5')

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while(True):
        ret, img = capture.read()
        # img = cv2.flip(img, 1)
        img = cv2.copyMakeBorder(img, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=0)
        draw_img = detect_face(retinaface_model, pfld_model, anchors, img)
        cv2.imshow('frame', draw_img)
        if cv2.waitKey(1) == ord('q'):
            break
