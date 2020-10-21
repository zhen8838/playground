import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.getcwd())
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from tools.auto_anno_movie import AnnoationBase, AnnoationSub, get_point_annotation, LANDMARKS
from easydict import EasyDict
import glob
import json
from typing import List, Dict, AnyStr, Any, Tuple
import argparse
plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Annotate(object):
  def __init__(self, anno: AnnoationBase, plabels: List[str]):
    self.anno = anno
    self.ax = plt.gca()
    self.points: List = []
    self.rects: List = []
    self.modes = ['Point', 'Face']
    self.delete = False
    self.nlandm = len(plabels)
    self.group_id: int = 0
    self.is_press = False
    self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
    self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
    self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
    self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    self.colors = ['b', 'g', 'r', 'c', 'm']
    self.point_idx = 0
    self.faces_idx = 0
    self.face_point_idx = 0

    """  根据group_ids构建对应数量的矩形框 """
    self.faces = [sub for sub in anno.shapes if sub.label == 'face']
    # sort face by group id
    self.faces: List[AnnoationSub] = sorted(self.faces, key=lambda sub: sub.group_id)

    for i in range(len(self.faces)):
      xmin, ymin = self.faces[i].points[0]
      xmax, ymax = self.faces[i].points[1]
      rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='b', fill=None)
      self.rects.append(rect)
      self.ax.add_patch(rect)
      text = plt.text(xmin, ymin, f'{self.faces[i].group_id:02d}', fontsize=20, c='r')

    """  根据group_ids构建对应数量的landmark """
    for i in range(len(self.faces)):
      for j in range(self.nlandm):
        circle = Circle((0, 0), 3, color=self.colors[j])
        self.points.append(circle)
        self.ax.add_patch(circle)
    """ 如果已经有点标注了，那么修正对应的landmark位置 """
    labelset = set([sub.label for sub in anno.shapes])
    if len(labelset) > 1:
      for i in range(len(self.faces)):
        for j, label in enumerate(plabels):
          # filter 找到对应id与label的注释
          point_anno: AnnoationSub = list(filter(
              lambda sub: (sub.group_id == self.faces[i].group_id
                           and sub.label == label), anno.shapes))[0]
          self.points[i * self.nlandm + j].set_center(tuple(point_anno['points'][0]))

    """ 初始化标题 """
    self.ax.set_title(f'点编辑模式  第{self.point_idx//self.nlandm}张脸  第{self.point_idx%self.nlandm}个点')

  def on_key_press(self, event):
    if event.key == 'x':
      self.modes.reverse()
    elif event.key == '-':
      if self.modes[0] == 'Point':
        self.sub_point_idx()
      else:
        self.sub_faces_idx()
    elif event.key == '+':
      if self.modes[0] == 'Point':
        self.add_point_idx()
      else:
        self.add_faces_idx()
    elif event.key == 'd':
      self.delete = True
      plt.close(self.ax.figure)
    self.update_title()

  def on_press(self, event):
    self.is_press = True

  def on_move(self, event):
    if self.is_press and self.modes[0] == 'Face':
      self.change_rect(event)
      self.update_title()

  def update_title(self):
    if self.modes[0] == 'Point':
      if self.point_idx >= len(self.points):
        self.ax.set_title('点已经分配完毕')
      else:
        self.ax.set_title(
            f'点编辑模式  第{self.point_idx//self.nlandm}张脸  第{self.point_idx%self.nlandm}个点')
    else:
      if self.faces_idx >= len(self.faces):
        self.ax.set_title('没有更多人脸了')
      else:
        self.ax.set_title(f'脸编辑模式  第{self.faces_idx}张脸 第{self.face_point_idx%2}个点')
    self.ax.figure.canvas.draw()

  def on_release(self, event):
    if self.modes[0] == 'Point':
      self.change_point(event)
      self.add_point_idx()
    else:
      self.change_rect(event)
      self.add_faces_idx()
    self.is_press = False
    self.update_title()

  def add_faces_idx(self):
    self.face_point_idx = min(2, self.face_point_idx + 1)
    if self.face_point_idx == 2:
      self.faces_idx = min(len(self.faces), self.faces_idx + 1)
      self.face_point_idx = 0

  def add_point_idx(self):
    self.point_idx = min(len(self.points), self.point_idx + 1)

  def sub_point_idx(self):
    self.point_idx = max(0, self.point_idx - 1)

  def sub_faces_idx(self):
    self.face_point_idx -= 1
    if self.face_point_idx < 0:
      if self.faces_idx >= 1:
        self.face_point_idx = 1
      self.faces_idx = max(0, self.faces_idx - 1)

  def change_rect(self, event):
    if self.faces_idx < len(self.faces):
      rect = self.rects[self.faces_idx]
      x, y = event.xdata, event.ydata
      if self.face_point_idx == 0:
        rect.set_xy((x, y))
      elif self.face_point_idx == 1:
        x0, y0 = rect.get_xy()
        rect.set_width(x - x0)
        rect.set_height(y - y0)

  def change_point(self, event):
    if self.point_idx < len(self.points):
      self.points[self.point_idx].set_center((event.xdata,
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
  offset = 0
  start_id = 0
  if start_item != '':
    start_id = json_list.index(f'{dataset_path}/{start_item}.json')

  for idx in range(start_id, number):
    json_path = json_list[idx - offset]
    anno = load_json(json_path)
    img_path = osp.join(dataset_path, anno.imagePath)
    ims = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(16, 9))
    plt.imshow(ims)
    handler = Annotate(anno, LANDMARKS)
    plt.show()

    for i, rect in enumerate(handler.rects):
      handler.faces[i].points = [[rect._x0, rect._y0],
                                 [rect._x1, rect._y1]]
      for j in range(5):
        point = handler.points[i * 5 + j]
        point_anno = get_point_annotation(*point.center,
                                          LANDMARKS[j],
                                          handler.faces[i].group_id)
        handler.anno.shapes.append(point_anno)

    if handler.delete is True:
      print("Delete", img_path)
      os.remove(img_path)
      os.remove(json_path)
      json_list.pop(idx - offset)
      offset += 1
    else:
      save_json(handler.anno, json_path)
      print("Rewrite", img_path)


if __name__ == "__main__":
  parse = argparse.ArgumentParser()
  parse.add_argument('--dataset', default='/media/zqh/Documents/jojo-face-landmark/01')
  parse.add_argument('--id', default='')
  args = parse.parse_args()
  main(args.dataset, args.id)
  """     
  python tools/manual_anno_movie.py  --dataset="/media/zqh/Documents/jojo-face-landmark/03"
  """