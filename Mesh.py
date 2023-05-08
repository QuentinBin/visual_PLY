'''
Description: mesh class
Author: Bin Peng
Date: 2023-05-04 14:09:06
LastEditTime: 2023-05-08 16:30:21
'''
from OpenGL.GL import *

class Vertex(object):
	def __init__(self) -> None:
		self._x = 0.0
		self._y = 0.0
		self._z = 0.0
		self._nx = 0.0
		self._ny = 0.0
		self._nz = 0.0
		self._s = 0.0
		self._t = 0.0
		self._hasCoords = False
		self._hasNormals = False
		self._hasST = False
	
	#----------coords----------#
	def set_X(self, x):
		self._hasCoords = True
		self._x = x
	
	def set_Y(self, y):
		self._hasCoords = True
		self._y = y
	
	def set_Z(self, z):
		self._hasCoords = True
		self._z = z
	
	def coords(self):
		return (self._x, self._y, self._z)
	
	def has_coords(self):
		return self._hasCoords
	
	#----------normals----------#
	def set_NX(self, nx):
		self._hasNormal = True
		self._nx = nx
	
	def set_NY(self, ny):
		self._hasNormal = True
		self._ny = ny
	
	def set_NZ(self, nz):
		self._hasNormal = True
		self._nz = nz
	
	def normal(self):
		return (self._nx, self._ny, self._nz)
	
	def has_normal(self):
		return self._hasNormal
	
	#----------ST----------#
	def set_S(self, s):
		self._hasST = True
		self._s = s

	def set_T(self, t):
		self._hasST = True
		self._t = t

	def stcoords(self):
		return (self._s, self._t)

	def has_ST(self):
		return self._hasST
	

class Face(object):
	'''
	note: A face has >= 3 vertices
	'''
	def __init__(self) -> None:
		self._vertices = []
	
	def set_vertices(self, l):
		self._vertices = l
	
	def vertices(self):
		return self._vertices
	

class Mesh(object):
	def	__init__(self) -> None:
		super(Mesh, self).__init__()
		self.vertices = []
		self.faces = []

	def add_vertex(self, v):
		self.vertices.append(v)

	def get_vertex(self, v_index):
		return self.vertices[v_index]
	
	def add_face(self, f):
		self.faces.append(f)




