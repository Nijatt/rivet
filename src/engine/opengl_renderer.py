import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL import GL
import numpy as np, math

# ← use the shared camera that already has handle_input etc.
from engine.gl_camera import Camera

# ─────────────────────────── Shaders ────────────────────────────
SPHERE_VS = """
#version 330 core
layout(location=0) in vec3 in_pos;
uniform mat4 u_model, u_view, u_proj;
void main(){ gl_Position = u_proj * u_view * u_model * vec4(in_pos,1.0); }
"""
SPHERE_FS = """
#version 330 core
uniform vec3 u_color;
out vec4 fragColor;
void main(){ fragColor = vec4(u_color,1.0); }
"""

LINE_VS = """
#version 330 core
layout(location=0) in vec3 in_pos;
uniform mat4 u_view, u_proj;
void main(){ gl_Position = u_proj * u_view * vec4(in_pos,1.0); }
"""
LINE_FS = """
#version 330 core
uniform vec3 u_lin_col;
out vec4 fragColor;
void main(){ fragColor = vec4(u_lin_col,1.0); }
"""

# ───────────────────── OpenGL helpers ───────────────────────────
def _compile(src, st):
    sid = GL.glCreateShader(st)
    GL.glShaderSource(sid, src)
    GL.glCompileShader(sid)
    if GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
        raise RuntimeError(GL.glGetShaderInfoLog(sid).decode())
    return sid

def _link(vs, fs):
    pid = GL.glCreateProgram()
    GL.glAttachShader(pid, vs); GL.glAttachShader(pid, fs)
    GL.glLinkProgram(pid)
    if GL.glGetProgramiv(pid, GL.GL_LINK_STATUS) != GL.GL_TRUE:
        raise RuntimeError(GL.glGetProgramInfoLog(pid).decode())
    GL.glDeleteShader(vs); GL.glDeleteShader(fs)
    return pid

# ───────────────────────── Sphere mesh ──────────────────────────
def make_uv_sphere(stacks=32, slices=48):
    verts, idx = [], []
    for i in range(stacks + 1):
        v = i / stacks
        th = v * math.pi
        st, ct = math.sin(th), math.cos(th)
        for j in range(slices + 1):
            u = j / slices
            ph = u * 2 * math.pi
            sp, cp = math.sin(ph), math.cos(ph)
            verts.append((st * cp, ct, st * sp))
    stride = slices + 1
    for i in range(stacks):
        for j in range(slices):
            a = i * stride + j
            b = a + stride
            idx += [a, b, a + 1, b, b + 1, a + 1]
    return np.array(verts, np.float32), np.array(idx, np.uint32)

# ───────────────────── OpenGLRenderer class ─────────────────────
class OpenGLRenderer:
    def __init__(self, w=960, h=600):
        pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
        self.w, self.h = w, h

        # ----- compile / link programs -----
        self.sph_prog = _link(_compile(SPHERE_VS, GL.GL_VERTEX_SHADER),
                              _compile(SPHERE_FS, GL.GL_FRAGMENT_SHADER))
        self.lin_prog = _link(_compile(LINE_VS, GL.GL_VERTEX_SHADER),
                              _compile(LINE_FS, GL.GL_FRAGMENT_SHADER))

        self.loc_sph = {n: GL.glGetUniformLocation(self.sph_prog, n)
                        for n in ("u_model", "u_view", "u_proj", "u_color")}
        self.loc_lin = {n: GL.glGetUniformLocation(self.lin_prog, n)
                        for n in ("u_view", "u_proj", "u_lin_col")}

        # ----- sphere geometry -----
        v, i = make_uv_sphere()
        self.sph_cnt = i.size
        self.sph_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.sph_vao)
        vbo = GL.glGenBuffers(1); GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)
        ebo = GL.glGenBuffers(1); GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)
        GL.glBindVertexArray(0)

        # ----- dynamic line buffer (up to 1024 segments) -----
        self.lin_vao = GL.glGenVertexArrays(1)
        self.lin_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.lin_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.lin_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 3 * 2 * 1024, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)
        GL.glBindVertexArray(0)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)

        # ----- shared camera with handle_input -----
        self.camera = Camera()

    # helper to build model matrix
    @staticmethod
    def _model(center, radius):
        # WRONG ↓                           dtype is in the wrong position
        # M = np.eye(4, np.float32)
        # RIGHT ↓
        M = np.eye(4, dtype=np.float32)
        M[0,0] = M[1,1] = M[2,2] = radius
        M[0,3], M[1,3], M[2,3] = center
        return M

    # main draw
    def render(self, bodies, lines=None):
        GL.glViewport(0, 0, self.w, self.h)
        GL.glClearColor(0.05, 0.06, 0.10, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(self.w / self.h)

        # ---- spheres ----
        GL.glUseProgram(self.sph_prog)
        GL.glUniformMatrix4fv(self.loc_sph["u_view"], 1, True, view)
        GL.glUniformMatrix4fv(self.loc_sph["u_proj"], 1, True, proj)
        GL.glBindVertexArray(self.sph_vao)
        for i, b in enumerate(bodies):
            GL.glUniformMatrix4fv(
                self.loc_sph["u_model"], 1, True,
                self._model(b.transform.position.astype(np.float32), float(b.radius))
            )
            t = (i * 37) % 100 / 100.0
            GL.glUniform3f(self.loc_sph["u_color"],
                           0.3 + 0.6 * t,
                           0.7 - 0.4 * t,
                           0.5 + 0.4 * (1 - t))
            GL.glDrawElements(GL.GL_TRIANGLES, self.sph_cnt, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

        # ---- lines ----
        if lines:
            verts = np.array([p for seg in lines for p in (seg[0], seg[1])], np.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.lin_vbo)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

            GL.glUseProgram(self.lin_prog)
            GL.glUniformMatrix4fv(self.loc_lin["u_view"], 1, True, view)
            GL.glUniformMatrix4fv(self.loc_lin["u_proj"], 1, True, proj)
            GL.glBindVertexArray(self.lin_vao)
            off = 0
            for seg in lines:
                col = seg[2] if len(seg) == 3 else (1.0, 1.0, 1.0)
                GL.glUniform3f(self.loc_lin["u_lin_col"], *col)
                GL.glDrawArrays(GL.GL_LINES, off, 2)
                off += 2
            GL.glBindVertexArray(0)

        pygame.display.flip()
