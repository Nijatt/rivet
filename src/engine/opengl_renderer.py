import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL import GL
import numpy as np, math

# ---------- Shaders ----------
VERT_SRC = """
#version 330 core
layout(location=0) in vec3 in_pos;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
void main(){
    gl_Position = u_proj * u_view * u_model * vec4(in_pos, 1.0);
}
"""
FRAG_SRC = """
#version 330 core
uniform vec3 u_color;
out vec4 fragColor;
void main(){
    fragColor = vec4(u_color, 1.0);
}
"""

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

def make_uv_sphere(stacks=32, slices=48):
    verts=[]; idx=[]
    for i in range(stacks+1):
        v=i/stacks; th=v*math.pi
        st,ct=math.sin(th),math.cos(th)
        for j in range(slices+1):
            u=j/slices; ph=u*2*math.pi
            sp,cp=math.sin(ph),math.cos(ph)
            x=st*cp; y=ct; z=st*sp
            verts.append((x,y,z))
    stride=slices+1
    for i in range(stacks):
        for j in range(slices):
            a=i*stride+j; b=a+stride
            idx += [a,b,a+1, b,b+1,a+1]
    return np.array(verts,np.float32), np.array(idx,np.uint32)

def _norm(v):
    n=np.linalg.norm(v)
    return v if n<1e-9 else v/n

class Camera:
    def __init__(self, position=(0,2,18), target=(0,1,0), up=(0,1,0),
                 fov_deg=60, z_near=0.1, z_far=200.0):
        self.position=np.array(position,dtype=np.float32)
        self.target  =np.array(target,dtype=np.float32)
        self.up      =np.array(up,dtype=np.float32)
        self.fov_deg=fov_deg; self.z_near=z_near; self.z_far=z_far

    def view_matrix(self):
        f = _norm(self.target - self.position)
        s = _norm(np.cross(f, self.up))
        u = np.cross(s, f)
        # Build standard row-major view: rows are (s, u, -f), translation last column
        view = np.eye(4, dtype=np.float32)
        view[0,0:3] = s
        view[1,0:3] = u
        view[2,0:3] = -f
        # translation components:
        view[0,3] = -np.dot(s, self.position)
        view[1,3] = -np.dot(u, self.position)
        view[2,3] =  np.dot(f, self.position)
        return view

    def projection_matrix(self, aspect):
        f=1.0/math.tan(math.radians(self.fov_deg)/2)
        zn,zf=self.z_near,self.z_far
        proj = np.array([
            [f/aspect,0,0,0],
            [0,f,0,0],
            [0,0,(zf+zn)/(zn - zf),(2*zf*zn)/(zn - zf)],
            [0,0,-1,0]
        ],dtype=np.float32)
        return proj

class OpenGLRenderer:
    def __init__(self, width=960, height=600):
        pygame.display.set_mode((width,height), DOUBLEBUF|OPENGL)
        self.width,self.height=width,height
        vs=_compile(VERT_SRC,GL.GL_VERTEX_SHADER)
        fs=_compile(FRAG_SRC,GL.GL_FRAGMENT_SHADER)
        self.program=_link(vs,fs)

        verts,indices=make_uv_sphere()
        self.index_count=indices.size

        self.vao=GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.vbo=GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER,verts.nbytes,verts,GL.GL_STATIC_DRAW)
        self.ebo=GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,0,None)
        GL.glBindVertexArray(0)

        self.loc_model=GL.glGetUniformLocation(self.program,"u_model")
        self.loc_view =GL.glGetUniformLocation(self.program,"u_view")
        self.loc_proj =GL.glGetUniformLocation(self.program,"u_proj")
        self.loc_color=GL.glGetUniformLocation(self.program,"u_color")

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)

        self.camera=Camera()

    def _model_matrix(self, center, radius):
        M=np.eye(4,dtype=np.float32)
        M[0,0]=M[1,1]=M[2,2]=radius
        M[0,3]=center[0]
        M[1,3]=center[1]
        M[2,3]=center[2]
        return M

    def render(self, bodies):
        GL.glViewport(0,0,self.width,self.height)
        GL.glClearColor(0.05,0.06,0.10,1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.program)

        view=self.camera.view_matrix()
        proj=self.camera.projection_matrix(self.width/self.height)

        # Upload with GL_TRUE (transpose) so our row-major matrices become correct column-major for GL
        GL.glUniformMatrix4fv(self.loc_view, 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(self.loc_proj, 1, GL.GL_TRUE, proj)

        GL.glBindVertexArray(self.vao)
        for i,b in enumerate(bodies):
            c = np.asarray(b.transform.position, dtype=np.float32)
            r = float(b.radius)
            model=self._model_matrix(c,r)
            GL.glUniformMatrix4fv(self.loc_model,1,GL.GL_TRUE,model)

            t=(i*37)%100/100.0
            GL.glUniform3f(self.loc_color, 0.3+0.7*t, 0.6-0.3*t, 0.4+0.5*(1-t))
            GL.glDrawElements(GL.GL_TRIANGLES,self.index_count,GL.GL_UNSIGNED_INT,None)

        GL.glBindVertexArray(0)
        pygame.display.flip()
