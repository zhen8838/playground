import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.getcwd())
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from tools.auto_anno_movie import AnnoationBase, AnnoationSub, get_landmark_annotation, get_face_annotation, LANDMARKS
from easydict import EasyDict
import glob
import json
from typing import List, Dict, AnyStr, Any, Tuple
import argparse
from enum import Enum
plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['keymap.save'].remove('s')
plt.rcParams['keymap.quit'].remove('q')


class RetrunState(Enum):
  delete = 1
  last = 2
  next = 3


class Annotate(object):

  def __init__(self, anno: AnnoationBase, plabels: List[str]):
    self.anno = anno
    self.ax = plt.gca()
    self.circles: List = []  # lanmark对应的matplotlib对象
    self.rects: List = []  # face对应的matplotlib对象
    self.modes = ['Point', 'Face']
    self.retrun = RetrunState.next
    self.nlandm = len(plabels)
    self.plabels = plabels
    self.group_id: int = 0
    self.is_press = False
    self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
    self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
    self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
    self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    self.colors = ['b', 'g', 'r', 'c', 'm']
    self.circles_idx = 0
    self.rects_idx = 0
    self.rect_point_idx = 0

    self.faces: List[AnnoationSub] = []  # 保存已有的人脸标注
    self.landmarks: List[AnnoationSub] = []  # 保存已有的坐标标注

    """  根据group_ids构建对应数量的矩形框 """
    self.faces = [sub for sub in anno.shapes if sub.label == 'face']
    # sort face by group id
    self.faces: List[AnnoationSub] = sorted(self.faces, key=lambda sub: sub.group_id)

    """ 绘制rect标注 """
    for i in range(len(self.faces)):
      xmin, ymin = self.faces[i].points[0]
      xmax, ymax = self.faces[i].points[1]
      rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='b', fill=None)
      self.rects.append(rect)
      self.ax.add_patch(rect)
      text = plt.text(xmin, ymin, f'{self.faces[i].group_id:02d}', fontsize=20, c='r')

    """ 绘制landmark标注 """
    for i in range(len(self.faces)):
      for j, label in enumerate(plabels):
        landmark_idx = self.find_landmark_index(anno.shapes, self.faces[i].group_id, label)
        landmark = anno.shapes[landmark_idx]
        circle = Circle((0, 0), 3, color=self.colors[j])
        circle.set_center(tuple(landmark['points'][0]))
        self.landmarks.append(landmark)
        self.circles.append(circle)
        self.ax.add_patch(circle)

    """ 初始化标题 """
    self.ax.set_title(
        f'点编辑模式  第{self.circles_idx//self.nlandm}张脸  第{self.circles_idx%self.nlandm}个点')

  @staticmethod
  def find_landmark_index(shapes: List[AnnoationBase], t_group_id: int, t_label: str):
    for idx, sub in enumerate(shapes):
      if (sub.group_id == t_group_id) and (sub.label == t_label):
        return idx
    return None

  def on_key_press(self, event):
    if event.key == 's':
      self.modes.reverse()
    elif event.key == 'q':
      if self.modes[0] == 'Point':
        self.sub_circle_idx()
      else:
        self.sub_rect_idx()
    elif event.key == 'e':
      if self.modes[0] == 'Point':
        self.add_circle_idx()
      else:
        self.add_rect_idx()
    elif event.key == 'w':
      self.retrun = RetrunState.delete
      plt.close(self.ax.figure)
    elif event.key == 'a':
      self.retrun = RetrunState.last
      plt.close(self.ax.figure)
    elif event.key == 'd':
      self.retrun = RetrunState.next
      plt.close(self.ax.figure)
    elif event.key == 'z':
      self.add_face(event)
    self.update_title()

  def add_face(self, event):
    """ 增加一个人脸 """
    x, y = event.xdata, event.ydata
    newface = EasyDict(get_face_annotation(x, y, x + 100, y + 100, len(self.faces)))
    self.anno.shapes.append(newface)
    self.faces.append(newface)
    rect = Rectangle((x, y), 100, 100, edgecolor='b', fill=None)
    self.rects.append(rect)
    self.ax.add_patch(rect)
    text = plt.text(x, y, f'{self.faces[-1].group_id:02d}', fontsize=20, c='r')

    """ 绘制landmark标注 """
    for j, label in enumerate(self.plabels):
      landmark = EasyDict(get_landmark_annotation(x + 50, y + 50, label, self.faces[-1].group_id))
      circle = Circle((x + 50, y + 50), 3, color=self.colors[j])
      self.anno.shapes.append(landmark)
      self.landmarks.append(landmark)
      self.circles.append(circle)
      self.ax.add_patch(circle)

  def on_press(self, event):
    self.is_press = True

  def on_move(self, event):
    if self.is_press and self.modes[0] == 'Face':
      self.change_rect(event)
      self.update_title()

  def update_title(self):
    if self.modes[0] == 'Point':
      if self.circles_idx >= len(self.circles):
        self.ax.set_title('点已经分配完毕')
      else:
        self.ax.set_title(
            f'点编辑模式  第{self.circles_idx//self.nlandm}张脸  第{self.circles_idx%self.nlandm}个点')
    else:
      if self.rects_idx >= len(self.rects):
        self.ax.set_title('没有更多人脸了')
      else:
        self.ax.set_title(f'脸编辑模式  第{self.rects_idx}张脸 第{self.rect_point_idx%2}个点')
    self.ax.figure.canvas.draw()

  def on_release(self, event):
    if self.modes[0] == 'Point':
      self.change_circle(event)
      self.add_circle_idx()
    else:
      self.change_rect(event)
      self.add_rect_idx()
    self.is_press = False
    self.update_title()

  def add_rect_idx(self):
    self.rect_point_idx = min(2, self.rect_point_idx + 1)
    if self.rect_point_idx == 2:
      self.rects_idx = min(len(self.rects), self.rects_idx + 1)
      self.rect_point_idx = 0

  def add_circle_idx(self):
    self.circles_idx = min(len(self.circles), self.circles_idx + 1)

  def sub_circle_idx(self):
    self.circles_idx = max(0, self.circles_idx - 1)

  def sub_rect_idx(self):
    self.rect_point_idx -= 1
    if self.rect_point_idx < 0:
      if self.rects_idx >= 1:
        self.rect_point_idx = 1
      self.rects_idx = max(0, self.rects_idx - 1)

  def change_rect(self, event):
    if self.rects_idx < len(self.rects):
      rect = self.rects[self.rects_idx]
      x, y = event.xdata, event.ydata
      if self.rect_point_idx == 0:
        rect.set_xy((x, y))
      elif self.rect_point_idx == 1:
        x0, y0 = rect.get_xy()
        rect.set_width(x - x0)
        rect.set_height(y - y0)

  def change_circle(self, event):
    if self.circles_idx < len(self.circles):
      self.circles[self.circles_idx].set_center((event.xdata,
                                                 event.ydata))


def load_json(json_path: str) -> AnnoationBase:
  with open(json_path, 'r+') as f:
    anno: AnnoationBase = EasyDict(json.load(f))
  return anno


def save_json(anno: AnnoationBase, json_path: str):
  with open(json_path, 'w') as f:
    text = json.dumps(anno, indent=4)
    f.write(text)


def main(dataset_path, start_item):
  # dataset_path = '/media/zqh/Documents/jojo-face-landmark/01'
  json_list = list(glob.glob(dataset_path + '/*.json'))
  number = len(json_list)
  idx = -1
  if start_item != '':
    idx = json_list.index(f'{dataset_path}/{start_item}.json')
  remove_list = []
  retrunstate = RetrunState.next

  while True:
    # get next
    if retrunstate == RetrunState.next:
      idx += 1
    elif retrunstate == RetrunState.last:
      idx -= 1

    if idx < 0 or idx >= number:
      print("Json files over")
      break

    if idx in remove_list:
      continue

    json_path = json_list[idx]
    print("Load", json_path)
    anno = load_json(json_path)
    img_path = osp.join(dataset_path, anno.imagePath)
    ims = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(16, 9))
    plt.imshow(ims)
    handler = Annotate(anno, LANDMARKS)
    plt.tight_layout(True)
    plt.show()

    # refush annotation
    for i, rect in enumerate(handler.rects):
      handler.faces[i].points = [[rect._x0, rect._y0],
                                 [rect._x1, rect._y1]]
    for i, circle in enumerate(handler.circles):
      x, y = circle.center
      handler.landmarks[i].points = [[x, y]]

    if handler.retrun == RetrunState.delete:
      print("Delete", img_path)
      os.remove(img_path)
      os.remove(json_path)
      remove_list.append(idx)
    else:
      retrunstate = handler.retrun
      save_json(handler.anno, json_path)
      print("Rewrite", json_path)


if __name__ == "__main__":
  parse = argparse.ArgumentParser()
  parse.add_argument('--dataset', default='/media/zqh/Documents/jojo-face-landmark/01')
  parse.add_argument('--id', default='')
  args = parse.parse_args()
  main(args.dataset, args.id)
  """     
  python tools/manual_anno_movie.py  --dataset="/media/zqh/Documents/jojo-face-landmark/03"
  """
