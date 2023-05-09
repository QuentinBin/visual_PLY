'''
Description: utils used for mesh
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-08 14:37:16
LastEditTime: 2023-05-09 12:32:49
'''
from OpenGL.GL import *

import Mesh


def draw_Mesh(mesh:Mesh.Mesh): 
	'''
	params
	------
	Mesh:class Mesh
	'''
	mode = None
	for face in mesh.faces:
		face:Mesh.Face
		num_vertices = len(face.vertices())
		if num_vertices == 3 and mode != GL_TRIANGLES:
			if mode:
				glEnd() # end last gl
			glBegin(GL_TRIANGLES)
			mode = GL_TRIANGLES
		elif num_vertices == 4 and mode != GL_QUADS:
			if mode:
				glEnd()
			glBegin(GL_QUADS)
			mode = GL_QUADS
		elif num_vertices > 4:
			if mode:
				glEnd()
			glBegin(GL_POLYGON)
			mode = GL_POLYGON
		elif num_vertices < 3:
			raise RuntimeError('Face has <3 vertices')
		for i in face.vertices():
			vertex:Mesh.Vertex = mesh.get_vertex(i)
			if vertex.has_normal():
				glNormal3f(*(vertex.normal()))
			glVertex3f(*(vertex.coords()))
		if mode == GL_POLYGON:
			glEnd()
			mode = None
	if mode:
		glEnd() # end final gl


def draw_MeshGrid(mesh:Mesh.Mesh):
	'''
	params
	------
	Mesh:class Mesh
	'''
	
	for face in mesh.faces:
		face:Mesh.Face
		num_vertices = len(face.vertices())
		if num_vertices >= 3:
			glBegin(GL_LINE_LOOP)
			for i in range(num_vertices):
				vertex_index = face.vertices()[i]
				# print(mesh.get_vertex(vertex_index).coords())
				glVertex3d(*(mesh.get_vertex(vertex_index).coords()))
			glEnd()


