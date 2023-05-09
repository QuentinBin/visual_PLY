'''
Description: None
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-09 09:29:44
LastEditTime: 2023-05-09 12:35:22
'''
import numpy as np
import sys
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import Mesh
import mesh_utils
import open_ply
import matrices
import draw_grasp



mesh = None
global cameraMatrix
global obj_distance_scale
cameraMatrix = matrices.getIdentity4x4()
obj_distance_scale = 1/20

scaleFactor = 0.95
rotateFactor = 0.05
translateFactor = 0.05

g_lightPos = (1.0, 1.0, 1.0, 0.0)


class draw_Grasp(object):
    '''
    params
    ------
    obj: mesh class
    gripper_confi: [gripper depth:float, radius:float]
    frame: str
    '''
    def __init__(self, obj:Mesh.Mesh, gripper_config=[0.1, 0.03], frame='object'):
        self._obj = obj
        self._frame = frame
        self._gripper_config = gripper_config # max_depth, radius
        self._grasp_axis = self.generate_grasp_axis()
        self._grasp_center = self.find_grasp_center()
        self._wrapped_faces_indices, self._unwrapped_faces_indices = [],[]
        self._wrapped_faces_indices, self._unwrapped_faces_indices = self.split_Mesh()
    
    def generate_grasp_axis(self):
        gripper_postion = np.random.random(3)-0.5
        gripper_postion = gripper_postion / np.linalg.norm(gripper_postion)
        return gripper_postion
    
    def find_grasp_center(self):
        min_projected_distance = float('inf')
        grasp_center = None
        for vertex in self._obj.vertices:
            vertex:Mesh.Vertex
            projected_distance = np.array(vertex.coords()).dot(self._grasp_axis)
            if projected_distance < min_projected_distance:
                grasp_center = np.array(vertex.coords())
                min_projected_distance = projected_distance
        return grasp_center

    
    def split_Mesh(self):
        grasp_depth = 0
        [max_depth, radius] = self._gripper_config
        while grasp_depth < max_depth:
            wrapped_faces_indices =  []
            unwrapped_faces_indices = []
            for face in self._obj.faces:
                unwrapped_faces_indices.append(face.vertices())
                face:Mesh.Face
                vertex_coords = self._obj.get_vertex(face.vertices()[0])
                vertex_coords = np.array(vertex_coords.coords())
                if abs((vertex_coords-self._grasp_center).dot(self._grasp_axis) - grasp_depth) < 0.0001:
                    # check collision
                    if np.linalg.norm(vertex_coords - (self._grasp_center+self._grasp_axis*grasp_depth)) > radius:
                        return wrapped_faces_indices, unwrapped_faces_indices
                elif (vertex_coords-self._grasp_center).dot(self._grasp_axis) < grasp_depth:
                    wrapped_faces_indices.append(unwrapped_faces_indices.pop())
                    
            grasp_depth += max_depth/20
        return wrapped_faces_indices, unwrapped_faces_indices
    
    def draw_MeshGrasp(self):
        # draw unwraped

        # print(self._unwrapped_faces_indices)
        print(self._wrapped_faces_indices)
        mode = None
        for face_vertices in self._unwrapped_faces_indices:
            glColor3f(1.0, 1.0, 1.0)
            num_vertices = len(face_vertices)
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
            for i in face_vertices:
                vertex:Mesh.Vertex = self._obj.get_vertex(i)
                if vertex.has_normal():
                    glNormal3f(*(vertex.normal()))
                glVertex3f(*(vertex.coords()))
            if mode == GL_POLYGON:
                glEnd()
                mode = None
        # draw wraped

        for face_vertices in self._wrapped_faces_indices:
            glColor3f(1.0, 0.0, 0.0)
            num_vertices = len(face_vertices)
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
            for i in face_vertices:
                vertex:Mesh.Vertex = self._obj.get_vertex(i)
                if vertex.has_normal():
                    glNormal3f(*(vertex.normal()))
                glVertex3f(*(vertex.coords()))
            if mode == GL_POLYGON:
                glEnd()
                mode = None
        if mode:
            glEnd() # end final gl




def doIdle():    
	pass



def do_Camera():
	global pos
	global obj_distance_scale
	global cameraMatrix
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	
	orientationMatrix = cameraMatrix.copy()
	orientationMatrix[3] = matrices.Vector4d(0, 0, 0, 1)
	pos_vector = matrices.Vector4d(0, 20*0.15, 20*0.5, 1)
	pos_vector.scale(obj_distance_scale)
	pos = pos_vector*cameraMatrix
	lookAt = matrices.Vector4d(0, 0, 0, 1)*cameraMatrix
	
	direction = matrices.Vector4d(0, 1, 0, 1)*orientationMatrix
	# print(pos.val)
	# print(lookAt.val)
	# print(direction.val)
	gluLookAt(*(pos.list()[:-1] + lookAt.list()[:-1] + direction.list()[:-1]))



def resize(width, height):
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	glViewport(0,0,width,height)
	gluPerspective(45.0, ((float)(width))/height, .1, 200)
	do_Camera()
	


def display():
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	do_Camera()
	glMatrixMode(GL_MODELVIEW)
	
	glColor3f(1, 1, 1)
	global draw
	draw.draw_MeshGrasp()
	glutSwapBuffers()



def doSpecial(*args):
	global cameraMatrix
	global obj_distance_scale
	if glutGetModifiers() & GLUT_ACTIVE_SHIFT:
		if args[0] == GLUT_KEY_UP:
			cameraMatrix = cameraMatrix*matrices.translate(0, -translateFactor, 0) #up
		if args[0] == GLUT_KEY_DOWN:
			cameraMatrix = cameraMatrix*matrices.translate(0, translateFactor, 0) #down
		if args[0] == GLUT_KEY_LEFT:
			cameraMatrix = cameraMatrix*matrices.translate(translateFactor, 0, 0) #left
		if args[0] == GLUT_KEY_RIGHT:
			cameraMatrix = cameraMatrix*matrices.translate(-translateFactor, 0, 0) #right
	else:
		if args[0] == GLUT_KEY_UP:
			cameraMatrix = cameraMatrix*matrices.rotateX(-rotateFactor) #up
		if args[0] == GLUT_KEY_DOWN:
			cameraMatrix = cameraMatrix*matrices.rotateX(rotateFactor) #down
		if args[0] == GLUT_KEY_LEFT:
			cameraMatrix = cameraMatrix*matrices.rotateY(-rotateFactor) #left
		if args[0] == GLUT_KEY_RIGHT:
			cameraMatrix = cameraMatrix*matrices.rotateY(rotateFactor) #right
		if args[0] == GLUT_KEY_F9:
			obj_distance_scale /= scaleFactor
			print("change scale to:",1/obj_distance_scale)
		elif args[0] == GLUT_KEY_F10:
			obj_distance_scale *= scaleFactor
			print("change scale to:",1/obj_distance_scale)
	display()



if __name__ == '__main__':

	mesh = open_ply.parse_ply(sys.argv[1])
	global draw
	draw = draw_grasp.draw_Grasp(mesh)
	draw.draw_MeshGrasp()
	
	glutInit([])
	glutInitDisplayMode( GLUT_RGB  | GLUT_DOUBLE | GLUT_DEPTH)
	glutInitWindowSize(640,480)
	glutCreateWindow("Simple OpenGL Renderer")
	glEnable(GL_DEPTH_TEST)      # Ensure farthest polygons render first
	glEnable(GL_NORMALIZE)       # Prevents scale from affecting color
	glClearColor(0.1, 0.1, 0.2, 0.0) 

	glutReshapeFunc(resize)
	glutDisplayFunc(display)
	glutIdleFunc(doIdle)      
	glutSpecialFunc(doSpecial)

	glShadeModel(GL_SMOOTH)

	glutMainLoop()