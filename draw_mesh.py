'''
Description: None
Author: Bin Peng
Date: 2023-05-08 21:43:47
LastEditTime: 2023-05-08 23:52:43
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

mesh = None
cameraMatrix = matrices.getIdentity4x4()
g_lightPos = (1.0, 1.0, 1.0, 0.0)
g_rotx = 20.0
g_roty = 30.0
g_rotz = 0.0
g_scale = 10.0

def doIdle():    
	pass

def do_Camera():
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	
	orientationMatrix = cameraMatrix.copy()
	orientationMatrix[3] = matrices.Vector4d(0, 0, 0, 1)
	pos = matrices.Vector4d(0, 20*0.15, 20*0.5, 1)*cameraMatrix
	lookAt = matrices.Vector4d(0, 0, 0, 1)*cameraMatrix
	direction = matrices.Vector4d(0, 1, 0, 1)*orientationMatrix
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
	mesh_utils.draw_MeshGrid(mesh)
	glutSwapBuffers()
		
def doKeyboard(*args):
	global cameraMatrix
	if args[0] == 'w':
		g_rotx += 5
	elif args[0] == 's':
		g_rotx -= 5
	elif args[0] == 'a':
		g_rotx += 5
	elif args[0] == 'd':
		g_rotx -= 5
	elif args[0] == 'q':
		g_rotz -= 5
	elif args[0] == 'e':
		g_rotx -= 5
	glutPostRedisplay()

if __name__ == '__main__':

	mesh = open_ply.parse_ply(sys.argv[1])
	
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
	glutKeyboardFunc(doKeyboard)
	# glutIdleFunc(idle);
	

	glEnable(GL_LIGHT0)
	glEnable(GL_LIGHTING)
	glLightfv(GL_LIGHT0, GL_POSITION, g_lightPos)

	# glShadeModel(GL_SMOOTH)

	glutMainLoop()