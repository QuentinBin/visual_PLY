'''
Description: 
Author: Bin Peng
Date: 2023-05-16 21:19:33
LastEditTime: 2023-05-16 23:45:46
'''
import numpy as np
import pymeshlab
from mayavi import mlab
import scipy.linalg as linalg
from scipy.spatial import Delaunay


from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D, CylinderPoint3D, ParallelJawPtGrasp3D



def load_mesh(path):
	# create a new MeshSet
	ms = pymeshlab.MeshSet()

	# load a new mesh in the MeshSet, and sets it as current mesh
	# the path of the mesh can be absolute or relative
	ms.load_new_mesh(path)

	# print("load mesh numbers:",len(ms))  # now ms contains 1 mesh
	
	# print("mesh names:", path)

	ms.set_current_mesh(0)

	# print("vertex number:", ms.current_mesh().vertex_number())

	return ms


class draw_Wrapped(object):
	'''
	params
	------
	obj: mesh class
	gripper_confi: [gripper depth:float, radius:float]
	frame: str
	'''
	def __init__(self, obj, sdf:GraspableObject3D, gripper_config=[0.1, 0.03], frame='object'):
		self._obj = obj
		self._sdf = sdf
		self._frame = frame
		self._gripper_radius = gripper_config[1]
		self._gripper_depth = gripper_config[0]
		self._vertex_num = self._obj.vertex_matrix().shape[0]
		self._gripper_config = gripper_config # max_depth, radius
		self._grasp_axis = self.generate_grasp_axis()
		self._grasp_center = self.find_grasp_center()
		self._grid_points = self.find_wrapped_region()
		print("points:",self._grid_points)
		
	
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
	
	def find_wrapped_region(self):
		dirs = 30 #12 directions
		r_steps = 10
		d_samples = 100
		points_in_list = []
		def rotate_mat(axis, radius):
			return linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radius))
		def cylinder_points(center, axis, depth, radius,convert_grid=True):          
			vertical_array = np.cross(axis, np.array([1,0,0]))
			vertical_array = vertical_array / np.linalg.norm(vertical_array)
			actions_points = np.zeros([d_samples, r_steps, dirs ,3])
			for d_i in range(actions_points.shape[0]):
				for r_i in range(actions_points.shape[1]):
					for i in range(actions_points.shape[2]):
						R = rotate_mat(axis, i*360/dirs*np.pi/180)
						actions_points[d_i,r_i, i,:] = center + axis * (d_i-d_samples/2)/d_samples *depth + vertical_array.dot(R) * r_i/r_steps*radius
						# mlab.points3d(actions_points[d_i,r_i, i,0],actions_points[d_i,r_i, i,1],actions_points[d_i,r_i, i,2],scale_factor=0.002,color = (1,0,0))
						if convert_grid:
							as_array = actions_points[d_i,r_i, i,:].T # 3x36
							# print(as_array.shape)
							transformed = self._sdf.sdf.transform_pt_obj_to_grid(as_array)
							actions_points[d_i,r_i, i,:] = transformed.T
						
			return actions_points
		

		actions_points = cylinder_points(self._grasp_center, self._grasp_axis, self._gripper_depth, self._gripper_radius)

		for r_i in range(actions_points.shape[1]):
			for i in range(actions_points.shape[2]):
				line_of_action = actions_points[:,r_i,i,:]
				# print(line_of_action)
				c_found, c = ParallelJawPtGrasp3D.find_contact(list(line_of_action), self._sdf, vis=False)
				if c_found:
					# print("find a contac!")
					points_in_list.append(c.point_)
					mlab.points3d(c.point_[0],c.point_[1],c.point_[2],scale_factor=0.003,color = (1,0,0))
		points_in_array = np.zeros([len(points_in_list),3])
		for i in range(len(points_in_list)):
			points_in_array[i,:] = points_in_list[i]
		return points_in_array
	
	
				

if __name__ == '__main__':
	ply_path = "/home/peng/桌面/visual_ply/mayavi_draw/simed_banana.ply"
	sdf_path = "/home/peng/桌面/visual_ply/mayavi_draw/nontextured.sdf"
	obj_path = "/home/peng/桌面/visual_ply/mayavi_draw/nontextured.obj"
	of = ObjFile(obj_path)
	sf = SdfFile(sdf_path)
	obj = GraspableObject3D(sf.read(), of.read())
	
	ms = load_mesh(ply_path)
	mesh = ms.current_mesh()
	# print(mesh.vertex_matrix())
	draw = draw_Wrapped(mesh, obj)
	print(draw._grasp_center)
	
	# mlab.pipeline.surface(mlab.pipeline.open(ply_path)) 
	
	# draw grid points
	mlab.points3d(draw._grasp_center[0],draw._grasp_center[1],draw._grasp_center[2],scale_factor=0.005)
	# draw trimesh
	tri = Delaunay(draw._grid_points)
	tri_index_matrix = tri.simplices
	for i in range(tri_index_matrix.shape[0]):
		x1,y1,z1 = draw._grid_points[tri_index_matrix[i,0],0], draw._grid_points[tri_index_matrix[i,0],1], draw._grid_points[tri_index_matrix[i,0],2]
		x2,y2,z2 = draw._grid_points[tri_index_matrix[i,1],0], draw._grid_points[tri_index_matrix[i,1],1], draw._grid_points[tri_index_matrix[i,1],2]
		x3,y3,z3 = draw._grid_points[tri_index_matrix[i,2],0], draw._grid_points[tri_index_matrix[i,2],1], draw._grid_points[tri_index_matrix[i,2],2]
		mlab.plot3d((x1,x2,x3),(y1,y2,y3),(z1,z2,z3),tube_radius = 0.002 ,color = (1,0,0))
	mlab.show()