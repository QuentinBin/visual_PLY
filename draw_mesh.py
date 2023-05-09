'''
Description: None
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-08 21:43:47
LastEditTime: 2023-05-09 12:30:39
'''

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
obj_distance_scale = 1/10

scaleFactor = 0.95
rotateFactor = 0.05
translateFactor = 0.05

g_lightPos = (1.0, 1.0, 1.0, 0.0)


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
	mesh_utils.draw_Mesh(mesh)
	# global draw
	# draw = draw_grasp.draw_Grasp(mesh)
	# draw.draw_MeshGrasp()
	
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