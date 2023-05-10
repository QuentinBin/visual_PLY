'''
Description: None
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-10 17:53:41
LastEditTime: 2023-05-10 18:04:58
'''
import open3d as o3d
mesh = o3d.io.read_triangle_mesh(r"/home/pengbin/桌面/visual_PLY/gripper_3Dmodels/Assem2.STL")
o3d.io.write_triangle_mesh(r"/home/pengbin/桌面/visual_PLY/gripper_3Dmodels/gripper.ply", mesh,write_ascii=True) #指定保存的类型