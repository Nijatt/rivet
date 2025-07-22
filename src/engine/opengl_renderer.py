import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL import GL
import numpy as np, math

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
    verts=[]; idx=[]
    for i in range(stacks+1):
        v=i/stacks; th=v*math.pi
        st,ct=math.sin(th),math.cos(th)
        for j in range(slices+1):
            u=j/slices; ph=u*2*math.pi
            sp,cp=math.sin(ph),math.cos(ph)
            verts.append((st*cp, ct, st*sp))
    stride=slices+1
    for i in range(stacks):
        for j in range(slices):
            a=i*stride+j; b=a+stride
            idx += [a,b,a+1, b,b+1,a+1]
    return np.array(verts,np.float32), np.array(idx,np.uint32)

# ───────────────────────── Camera ───────────────────────────────
def _norm(v): n=np.linalg.norm(v); return v if n<1e-9 else v/n
class Camera:
    def __init__(self,pos=(0,2,18),tgt=(0,1,0),up=(0,1,0),fov=60,zn=0.1,zf=200):
        self.pos=np.array(pos,float); self.tgt=np.array(tgt,float); self.up=np.array(up,float)
        self.fov=fov; self.zn=zn; self.zf=zf
    def view(self):
        f=_norm(self.tgt-self.pos); s=_norm(np.cross(f,self.up)); u=np.cross(s,f)
        V=np.eye(4,dtype=np.float32); V[0,:3]=s; V[1,:3]=u; V[2,:3]=-f
        V[0,3]=-np.dot(s,self.pos); V[1,3]=-np.dot(u,self.pos); V[2,3]=np.dot(f,self.pos)
        return V
    def proj(self,aspect):
        f=1/math.tan(math.radians(self.fov)/2); zn,zf=self.zn,self.zf
        return np.array([[f/aspect,0,0,0],[0,f,0,0],[0,0,(zf+zn)/(zn-zf),(2*zf*zn)/(zn-zf)],
                         [0,0,-1,0]],dtype=np.float32)

# ───────────────────── OpenGLRenderer class ─────────────────────
class OpenGLRenderer:
    def __init__(self,w=960,h=600):
        pygame.display.set_mode((w,h),DOUBLEBUF|OPENGL)
        self.w,self.h=w,h

        # sphere program
        self.sph_prog=_link(_compile(SPHERE_VS,GL.GL_VERTEX_SHADER),
                            _compile(SPHERE_FS,GL.GL_FRAGMENT_SHADER))
        self.loc_sph = {name:GL.glGetUniformLocation(self.sph_prog,name)
                        for name in ("u_model","u_view","u_proj","u_color")}

        # line program
        self.lin_prog=_link(_compile(LINE_VS,GL.GL_VERTEX_SHADER),
                            _compile(LINE_FS,GL.GL_FRAGMENT_SHADER))
        self.loc_lin = {name:GL.glGetUniformLocation(self.lin_prog,name)
                        for name in ("u_view","u_proj","u_lin_col")}

        # sphere VAO/VBO/EBO
        verts,idx=make_uv_sphere(); self.sph_count=idx.size
        self.sph_vao=GL.glGenVertexArrays(1); GL.glBindVertexArray(self.sph_vao)
        vbo=GL.glGenBuffers(1); GL.glBindBuffer(GL.GL_ARRAY_BUFFER,vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER,verts.nbytes,verts,GL.GL_STATIC_DRAW)
        ebo=GL.glGenBuffers(1); GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER,idx.nbytes,idx,GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,False,0,None)
        GL.glBindVertexArray(0)

        # dynamic line VAO/VBO (allocate once, update per frame)
        self.lin_vao=GL.glGenVertexArrays(1); self.lin_vbo=GL.glGenBuffers(1)
        GL.glBindVertexArray(self.lin_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,self.lin_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4*3*2*1024, None, GL.GL_DYNAMIC_DRAW) # up to 1024 lines
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,False,0,None)
        GL.glBindVertexArray(0)

        GL.glEnable(GL.GL_DEPTH_TEST); GL.glDisable(GL.GL_CULL_FACE)
        self.camera=Camera()

    # ---------- helpers ----------
    @staticmethod
    def _model(center,radius):
        M=np.eye(4,dtype=np.float32)
        M[0,0]=M[1,1]=M[2,2]=radius; M[0,3],M[1,3],M[2,3]=center
        return M

    # ---------- public render ----------
    def render(self, bodies, lines=None):
        GL.glViewport(0,0,self.w,self.h)
        GL.glClearColor(0.05,0.06,0.10,1); GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
        view=self.camera.view(); proj=self.camera.proj(self.w/self.h)

        # Draw spheres
        GL.glUseProgram(self.sph_prog)
        GL.glUniformMatrix4fv(self.loc_sph["u_view"],1,True,view)
        GL.glUniformMatrix4fv(self.loc_sph["u_proj"],1,True,proj)
        GL.glBindVertexArray(self.sph_vao)
        for i,b in enumerate(bodies):
            model=self._model(b.transform.position.astype(np.float32), float(b.radius))
            GL.glUniformMatrix4fv(self.loc_sph["u_model"],1,True,model)
            hue=(i*37)%100/100
            GL.glUniform3f(self.loc_sph["u_color"],0.3+0.6*hue,0.7-0.4*hue,0.5+0.4*(1-hue))
            GL.glDrawElements(GL.GL_TRIANGLES,self.sph_count,GL.GL_UNSIGNED_INT,None)
        GL.glBindVertexArray(0)

        # Draw lines if provided
        if lines:
            # flatten vertices
            verts=np.array([p for seg in lines for p in (seg[0],seg[1])],dtype=np.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER,self.lin_vbo)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER,0,verts.nbytes,verts)
            GL.glUseProgram(self.lin_prog)
            GL.glUniformMatrix4fv(self.loc_lin["u_view"],1,True,view)
            GL.glUniformMatrix4fv(self.loc_lin["u_proj"],1,True,proj)
            GL.glBindVertexArray(self.lin_vao)
            offset=0
            for seg in lines:
                col=np.array(seg[2] if len(seg)==3 else (1,1,1),dtype=np.float32)
                GL.glUniform3fv(self.loc_lin["u_lin_col"],1,col)
                GL.glDrawArrays(GL.GL_LINES,offset,2)
                offset+=2
            GL.glBindVertexArray(0)

        pygame.display.flip()
