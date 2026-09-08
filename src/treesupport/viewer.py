"""Persistent OpenGL mesh viewport; camera changes never rebuild mesh geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import sys
import time
import tkinter as tk

import numpy as np
from OpenGL import GL, GLU
from pyopengltk import OpenGLFrame


@dataclass
class Camera:
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    elevation: float = 25.
    azimuth: float = -55.
    radius: float = 10.
    half_height: float = 12.
    perspective: bool = False

    @property
    def distance(self):
        return self.half_height / np.tan(np.radians(20.))

    def basis(self):
        az, el = np.radians([self.azimuth, self.elevation])
        backward = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        right = np.array([-np.sin(az), np.cos(az), 0.])
        return right, np.cross(backward, right), backward

    def fit(self, bounds, aspect=1.):
        self.target = np.asarray(bounds, dtype=float).mean(axis=0)
        self.radius = max(float(np.linalg.norm(np.diff(bounds, axis=0)))/2, .001)
        self.half_height = self.radius * 1.15 / min(1., max(.01, aspect))

    def zoom(self, steps):
        self.half_height = float(np.clip(self.half_height * 1.15**(-np.clip(steps, -20, 20)),
                                         self.radius*.001, self.radius*1000))

    def pan(self, dx, dy, viewport_height):
        right, up, _ = self.basis()
        self.target += (-dx*right + dy*up) * (2*self.half_height/max(1, viewport_height))

    def orbit(self, dx, dy):
        self.azimuth = (self.azimuth - dx*.45) % 360
        self.elevation = float(np.clip(self.elevation + dy*.45, -89.95, 89.95))


class MeshViewport(OpenGLFrame):
    def __init__(self, master, **kwargs):
        self.camera = Camera()
        self.bounds = np.array([[-10., -10., 0.], [10., 10., 20.]])
        self.meshes = []
        self._buffers = []
        self._mesh_key = None
        self._dirty = True
        self._frame_job = None
        self._disposed = False
        self._drag = None
        self._fit_on_resize = True
        self.wireframe = False
        self.show_grid = True
        self.cutaway = False
        self.cut_x = 0.
        self.frame_times = []
        self.upload_count = 0
        self.renderer = ""
        self.error = None
        self._dc = self._rc = None
        super().__init__(master, width=650, height=450, takefocus=True, **kwargs)
        self.bind("<ButtonPress-1>", self._begin_drag)
        self.bind("<ButtonPress-2>", self._begin_drag)
        self.bind("<ButtonPress-3>", self._begin_drag)
        for button in (1, 2, 3):
            self.bind(f"<B{button}-Motion>", self._motion)
            self.bind(f"<ButtonRelease-{button}>", self._end_drag)
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self.zoom(1))
        self.bind("<Button-5>", lambda e: self.zoom(-1))
        for key in ("f", "F", "Home"):
            self.bind(f"<{key}>", lambda e: self.fit())
        for key in ("plus", "equal", "KP_Add"):
            self.bind(f"<{key}>", lambda e: self.zoom(1))
        for key in ("minus", "KP_Subtract"):
            self.bind(f"<{key}>", lambda e: self.zoom(-1))
        self.bind("<Left>", lambda e: self.orbit(-10, 0))
        self.bind("<Right>", lambda e: self.orbit(10, 0))
        self.bind("<Up>", lambda e: self.orbit(0, -10))
        self.bind("<Down>", lambda e: self.orbit(0, 10))
        self.bind("<Destroy>", self._destroy_event, add="+")

    def tkCreateContext(self):
        if sys.platform == "win32":
            from pyopengltk.win32 import pfd
            pfd.cDepthBits = 24
        super().tkCreateContext()
        if sys.platform == "win32":
            from OpenGL import WGL
            self._dc, self._rc = WGL.wglGetCurrentDC(), WGL.wglGetCurrentContext()

    def tkMap(self, event):
        if self._disposed or self.error:
            return
        try:
            super().tkMap(event)
            self.request_render()
        except Exception as exc:
            self.error = str(exc)
            tk.Label(self, text=f"OpenGL表示を初期化できません。\n{exc}", wraplength=400,
                     bg="#111820", fg="#e4edf4").place(relx=.5, rely=.5, anchor="center")

    def initgl(self):
        if not GL.glGetString(GL.GL_VERSION):
            raise RuntimeError("OpenGL context is unavailable")
        if not GL.glGenBuffers:
            raise RuntimeError("OpenGL 1.5以降の頂点バッファ対応ドライバーが必要です。")
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
        GL.glClearColor(17/255, 24/255, 32/255, 1.)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
        GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, [.28, .28, .28, 1])
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [.8, .8, .8, 1])

    def tkResize(self, event):
        self.width, self.height = max(1, event.width), max(1, event.height)
        if self._fit_on_resize:
            self.camera.fit(self.bounds, self.width/self.height)
        self.request_render()

    def tkExpose(self, event):
        self.request_render()

    def request_render(self):
        if not self._disposed and self._frame_job is None:
            self._frame_job = self.after(16, self._render_frame)

    def _render_frame(self):
        self._frame_job = None
        if self._disposed or self.error or not self.context_created or not self.winfo_ismapped():
            return
        start = time.perf_counter()
        self.tkMakeCurrent()
        self.redraw()
        self.tkSwapBuffers()
        self.frame_times.append(time.perf_counter()-start)
        self.frame_times = self.frame_times[-240:]

    def set_meshes(self, meshes, reset_camera=False):
        # Retain mesh references; generated models are replaced, not edited in place.
        key = tuple((id(mesh), color) for mesh, color in meshes)
        if key != self._mesh_key:
            self.meshes = list(meshes)
            self._mesh_key = key
            self._dirty = True
        if meshes:
            bounds = np.vstack([m.bounds for m, _ in meshes])
            self.bounds = np.array([bounds.min(axis=0), bounds.max(axis=0)])
        if reset_camera:
            self.camera.elevation, self.camera.azimuth = 25., -55.
            self.fit()
        self.request_render()

    def fit(self):
        self._fit_on_resize = True
        self.camera.fit(self.bounds, max(1, self.winfo_width())/max(1, self.winfo_height()))
        self.request_render()

    def set_camera(self, elevation, azimuth):
        self.camera.elevation, self.camera.azimuth = elevation, azimuth
        self.request_render()

    def zoom(self, steps):
        self._fit_on_resize = False
        self.camera.zoom(steps)
        self.request_render()
        return "break"

    def orbit(self, dx, dy):
        self.camera.orbit(dx, dy)
        self.request_render()

    def _wheel(self, event):
        self.focus_set()
        return self.zoom(event.delta/120)

    def _begin_drag(self, event):
        self.focus_set()
        self._drag = (event.x, event.y, event.num, bool(event.state & 1))

    def _end_drag(self, event):
        self._drag = None

    def _motion(self, event):
        if self._drag is None:
            return
        x, y, button, shifted = self._drag
        dx, dy = event.x-x, event.y-y
        if button in (2, 3) or shifted:
            self._fit_on_resize = False
            self.camera.pan(dx, dy, self.winfo_height())
        else:
            self.camera.orbit(dx, dy)
        self._drag = (event.x, event.y, button, shifted)
        self.request_render()

    def _upload(self):
        self._release_buffers()
        for mesh, color in self.meshes:
            data = np.empty((len(mesh.faces)*3, 6), dtype=np.float32)
            data[:, :3] = mesh.triangles.reshape(-1, 3)
            data[:, 3:] = np.repeat(mesh.face_normals, 3, axis=0)
            buffer = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            rgb = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
            self._buffers.append((buffer, len(data), rgb))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.upload_count += 1
        self._dirty = False

    def _release_buffers(self):
        for buffer, _, _ in self._buffers:
            GL.glDeleteBuffers(1, [buffer])
        self._buffers.clear()

    def redraw(self):
        if self._dirty:
            self._upload()
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        camera = self.camera
        near = max(camera.radius*.0001, camera.distance - camera.radius*4)
        far = camera.distance + camera.radius*6
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        if camera.perspective:
            GLU.gluPerspective(40, width/height, near, far)
        else:
            h = camera.half_height
            GL.glOrtho(-h*width/height, h*width/height, -h, h, -far, far)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [-.4, .7, 1., 0.])
        right, up, backward = camera.basis()
        eye = camera.target + backward*camera.distance
        GLU.gluLookAt(*eye, *camera.target, *up)
        if self.cutaway:
            GL.glClipPlane(GL.GL_CLIP_PLANE0, [-1., 0., 0., self.cut_x])
            GL.glEnable(GL.GL_CLIP_PLANE0)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        if self.wireframe:
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1, 1)
        for buffer, count, color in self._buffers:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer)
            GL.glVertexPointer(3, GL.GL_FLOAT, 24, ctypes.c_void_p(0))
            GL.glNormalPointer(GL.GL_FLOAT, 24, ctypes.c_void_p(12))
            GL.glColor3f(*color)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, count)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glDisable(GL.GL_LIGHTING)
        if self.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glColor3f(.18, .28, .30)
            for buffer, count, _ in self._buffers:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer)
                GL.glVertexPointer(3, GL.GL_FLOAT, 24, ctypes.c_void_p(0))
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, count)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glDisable(GL.GL_CLIP_PLANE0)
        if self.show_grid:
            self._grid()
        self._orientation(width, height)

    def _grid(self):
        center = self.bounds.mean(axis=0)
        extent = max(float(np.max(self.bounds[1]-self.bounds[0])), 1.)*.7
        GL.glColor3f(.18, .25, .31)
        GL.glBegin(GL.GL_LINES)
        for offset in np.linspace(-extent, extent, 15):
            GL.glVertex3f(center[0]-extent, center[1]+offset, 0)
            GL.glVertex3f(center[0]+extent, center[1]+offset, 0)
            GL.glVertex3f(center[0]+offset, center[1]-extent, 0)
            GL.glVertex3f(center[0]+offset, center[1]+extent, 0)
        GL.glEnd()

    def _orientation(self, width, height):
        GL.glViewport(12, 12, 72, 72)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1.4, 1.4, -1.4, 1.4, -5, 5)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        _, up, backward = self.camera.basis()
        GLU.gluLookAt(*(backward*3), 0, 0, 0, *up)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)
        for axis, color in zip(np.eye(3), [(1., .35, .35), (.4, .9, .4), (.4, .65, 1.)]):
            GL.glColor3f(*color)
            GL.glVertex3f(0, 0, 0)
            GL.glVertex3f(*axis)
        GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glViewport(0, 0, width, height)

    def save_image(self, path):
        from PIL import Image
        self.tkMakeCurrent()
        self.redraw()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        width, height = self.winfo_width(), self.winfo_height()
        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        Image.frombytes("RGB", (width, height), pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path)

    def _destroy_event(self, event):
        if event.widget is self:
            self.dispose()

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._frame_job is not None:
            self.after_cancel(self._frame_job)
            self._frame_job = None
        if self.context_created or self._rc:
            if sys.platform == "win32" and self._rc:
                from OpenGL import WGL
                WGL.wglMakeCurrent(self._dc, self._rc)
                self._release_buffers()
                WGL.wglMakeCurrent(None, None)
                WGL.wglDeleteContext(self._rc)
                release_dc = ctypes.windll.user32.ReleaseDC
                release_dc.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                release_dc(self._wid, self._dc)
                self._rc = None
            else:
                self.tkMakeCurrent()
                self._release_buffers()
        self.meshes.clear()
