import mss
from mss.linux import MSS
import mss.tools
import wx
import argparse
import os

# NOTE: conda install wxpython, pip install mss


class SelectableFrame(wx.Frame):

  c = None

  def __init__(self, w, h, screen_cat, save_pattern, number,
               parent=None, id=-1, title=""):
    wx.Frame.__init__(self, parent, id, title, size=wx.DisplaySize())

    self.panel = wx.Panel(self, size=self.GetSize())

    self.panel.Bind(wx.EVT_MOTION, self.OnMouseMove)
    self.panel.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
    self.panel.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
    self.panel.Bind(wx.EVT_PAINT, self.OnPaint)
    self.panel.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

    self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))

    self.SetTransparent(200)

    self.screen_cat: MSS = screen_cat
    self.old_w = w
    self.old_h = h
    self.w = self.old_w
    self.h = self.old_h
    self.half_w = self.old_w // 2
    self.half_h = self.old_h // 2
    self.roi_w = int(self.old_w * 0.7)
    self.roi_h = int(self.old_h * 0.7)
    self.roi_half_w = self.roi_w // 2
    self.roi_half_h = self.roi_h // 2
    self.save_pattern = save_pattern
    self.number = number

  def OnMouseMove(self, event):
    self.c = event.GetPosition()
    self.Refresh()

  def OnMouseDown(self, event):
    self.SetTransparent(0)
    self.c = event.GetPosition()
    self.Refresh()

  def OnMouseUp(self, event):
    print(self.c, self.ClientToScreen(self.c))
    wx.MicroSleep(200)
    cc = self.ClientToScreen(self.c)
    im = self.screen_cat.grab({"top": cc.y - self.half_h,    # offset from the top
                               "left": cc.x - self.half_w,  # offset from the left
                               "width": self.w,
                               "height": self.h})

    mss.tools.to_png(im.rgb, im.size, output=self.save_pattern % self.number)
    self.number += 1
    self.w = self.old_w
    self.h = self.old_h
    self.half_w = self.old_w // 2
    self.half_h = self.old_h // 2
    self.roi_w = int(self.old_w * 0.7)
    self.roi_h = int(self.old_h * 0.7)
    self.roi_half_w = self.roi_w // 2
    self.roi_half_h = self.roi_h // 2 
    self.SetTransparent(200)
    self.Refresh()

  def OnPaint(self, event):
    if self.c is None:
      return
    dc = wx.PaintDC(self.panel)
    dc.SetPen(wx.Pen('red', 3))
    brush = wx.TRANSPARENT_BRUSH
    dc.SetBrush(brush)
    dc.DrawRectangle(self.c.x - self.half_w, self.c.y - self.half_h, self.w, self.h)
    dc.DrawRectangle(self.c.x - self.roi_half_w, self.c.y - self.roi_half_h, self.roi_w, self.roi_h)

  def OnKeyDown(self, event):
    kc = event.GetKeyCode()
    if kc == wx.WXK_ESCAPE:  # esc
      self.Destroy()
    elif kc == wx.WXK_UP:
      self.h += 32
      self.w += 32
      self.half_h += 16
      self.half_w += 16
      self.roi_w += 22
      self.roi_h += 22
      self.roi_half_w += 11
      self.roi_half_h += 11
      self.Refresh()
    elif kc == wx.WXK_DOWN:
      self.h -= 32
      self.w -= 32
      self.half_h -= 16
      self.half_w -= 16
      self.roi_w -= 22
      self.roi_h -= 22
      self.roi_half_w -= 11
      self.roi_half_h -= 11
      self.Refresh()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir',
                      type=str,
                      help='savr dir',
                      default='ep1')
  parser.add_argument('--wh',
                      nargs='+',
                      type=int,
                      help='crop w h',
                      default=[512, 512])
  args = parser.parse_args()

  with mss.mss(display="") as sct:
    app = wx.App(0)
    w, h = args.wh
    save_pattern = args.dir + "/%d.png"
    if not os.path.exists(args.dir):
      os.mkdir(args.dir)
      number = 0
    else:
      l = os.listdir(args.dir)
      nl = [int(p.split('.')[0]) for p in l]
      if len(nl) > 0:
        number = max(nl) + 1
      else:
        number = 0

    frame = SelectableFrame(w, h, sct, save_pattern, number)

    frame.Show(True)
    app.SetTopWindow(frame)
    app.MainLoop()
