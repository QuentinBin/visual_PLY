'''
Description: test to draw a point in a mesh
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-15 10:51:51
LastEditTime: 2023-05-16 22:34:00
'''
import numpy as np
import pymeshlab
from mayavi import mlab


def load_mesh(path):
	# create a new MeshSet
	ms = pymeshlab.MeshSet()

	# load a new mesh in the MeshSet, and sets it as current mesh
	# the path of the mesh can be absolute or relative
	ms.load_new_mesh(path)

	print("load mesh numbers:",len(ms))  # now ms contains 1 mesh
	
	print("mesh names:", path)

	# set the first mesh (id 0) as current mesh
	ms.set_current_mesh(0)

	# print the number of vertices of the current mesh
	print("vertex number:", ms.current_mesh().vertex_number())

	return ms


class draw_Grasp(object):
	'''
	params
	------
	obj: mesh class
	gripper_confi: [gripper depth:float, radius:float]
	frame: str
	'''
	def __init__(self, obj, gripper_config=[0.1, 0.03], frame='object'):
		self._obj = obj
		self._frame = frame
		self._vertex_num = self._obj.vertex_matrix().shape[0]
		self._gripper_config = gripper_config # max_depth, radius
		self._grasp_axis = self.generate_grasp_axis()
		self._grasp_center = self.find_grasp_center()
		
	
	def generate_grasp_axis(self):
		gripper_postion = np.random.random(3)-0.5
		gripper_postion = gripper_postion / np.linalg.norm(gripper_postion)
		return gripper_postion
	
	def find_grasp_center(self):
		min_projected_distance = float('inf')
		grasp_center = None
		for vertex_index in range(self._vertex_num):
			vertex_coords = self._obj.vertex_matrix()[vertex_index]
			projected_distance = vertex_coords.dot(self._grasp_axis)
			if projected_distance < min_projected_distance:
				grasp_center = vertex_coords
				min_projected_distance = projected_distance
		return grasp_center


if __name__ == '__main__':
	ply_path = "/home/peng/桌面/visual_ply/mayavi_draw/simed_banana.ply"
	ms = load_mesh(ply_path)
	mesh = ms.current_mesh()
	print(mesh.vertex_matrix())
	draw = draw_Grasp(mesh)
	print(draw._grasp_center)
	
	mlab.pipeline.surface(mlab.pipeline.open(ply_path)) 
	mlab.points3d(draw._grasp_center[0],draw._grasp_center[1],draw._grasp_center[2],scale_factor=0.005)
	mlab.show()
