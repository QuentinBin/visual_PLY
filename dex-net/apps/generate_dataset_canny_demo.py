import numpy as np
import random
from scipy import linalg
import os
import matplotlib.pyplot as plt
from dexnet.grasping import Contact3D,GraspableObject3D
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import PointGraspMetrics3D
#
class Heatmap_Generate(object):
    """
        `input`
        Class Sdf3D: obj
        list: camera_matrix-----[f, dx, dy, u0, v0]
        list: gripper_config-----[max_depth, radius]
    """
    def __init__(self, obj, camera_matrix, gripper_config, frame='object'):
        self.obj_ = obj
        self.frame_ = frame
        self.picture_size_ = [400,400]
        self.surfpoints_in_obj_, _ = self.obj_.sdf.surface_points(grid_basis=False)
        self.camera_matrix_ = camera_matrix # f, dx, dy, u0, v0
        self.gripper_config_ = gripper_config # max_depth, radius
        self.T_table_in_obj_ = self.generate_T_table_in_obj()
        self.T_camera_in_table_ = self.generate_T_camera_in_table()
        self.surfpoints_in_camera_  = self.surfpoints_in_camera() # shape:3
        self.depth_img_, self.index_img_ = self.generate_depth()
        self.quality_img_, self.max_quality_ = self.generate_quality()

    def generate_T_table_in_obj(self):
        # generate table frame, which only has a rotaion about obj frame
        rotate_axis = np.random.random(3) - 0.5
        rotate_angle = random.random()*2*np.pi
        rotate_axis = rotate_axis / np.linalg.norm(rotate_axis)
        R = linalg.expm(np.cross(np.eye(3), rotate_axis / linalg.norm(rotate_axis) * rotate_angle))
        T_table_in_obj = np.eye(4)
        T_table_in_obj[:3,:3] = R

        return T_table_in_obj

    def generate_T_camera_in_table(self):
        """
        generate a camera pose at 1 m distance, whose Z_axis point at obj's origin, X_axis parallel to Table's
        
        return
        ------
        1. surfpoints_in_camera
        2. surfpoints_distance_in_camera
        """
        distance = 0.3
        P_camera_in_table = np.random.random(3) # 3X1 
        P_camera_in_table = P_camera_in_table / np.linalg.norm(P_camera_in_table) * distance

        z_axis = -P_camera_in_table / distance
        x_axis = self.T_table_in_obj_[0:3,0]
        y_axis = np.cross(z_axis, x_axis)

        T_camera_in_table = np.eye(4)
        T_camera_in_table[:3,0] = x_axis
        T_camera_in_table[:3,1] = y_axis
        T_camera_in_table[:3,2] = z_axis
        T_camera_in_table[:3,3] = P_camera_in_table

        return T_camera_in_table

    def surfpoints_in_camera(self):
        """
        `return`
        array 3: surfpoints_in_camera
        """
        T_obj_in_camera = np.eye(4)
        surfpoints_in_camera = []

        T_camera_in_obj = np.dot(self.T_table_in_obj_, self.T_camera_in_table_)
        P_camera_in_obj = T_camera_in_obj[:3,3]
        R_obj_in_camera = T_camera_in_obj[:3, :3].T
        T_obj_in_camera[:3, :3] = R_obj_in_camera
        T_obj_in_camera[:3, 3] = -R_obj_in_camera.dot(P_camera_in_obj)

        for x_surf in self.surfpoints_in_obj_:
            x_surf_in_obj = np.array([[x_surf[0]],[x_surf[1]],[x_surf[2]],[1]]) # 4X1
            x_surf_in_camera = T_obj_in_camera.dot(x_surf_in_obj) # shape:4X1
            x_surf_in_camera = x_surf_in_camera[0:3,0] # shape:3
            surfpoints_in_camera.append(x_surf_in_camera)

        return surfpoints_in_camera

    def generate_depth(self):
        """
        `return`
        1. np.array(picture_size):depth_image
        2. np.array(picture_size):index_image
        """
        depth_image = np.zeros(self.picture_size_)
        index_image = np.zeros(self.picture_size_,dtype=int) - 1 # store the index of surfpoints

        [f, dx, dy, u0, v0] = self.camera_matrix_

        for index, x_surf in enumerate(self.surfpoints_in_camera_):
            u = int( f/dx * (x_surf[0]/x_surf[2]) + u0 )
            v = int( f/dy * (x_surf[1]/x_surf[2]) + v0 )
            if depth_image[u,v] > x_surf[2] or abs(depth_image[u,v])<1e-5: # only store the closer one if points coincide
                depth_image[u,v] = x_surf[2]
                index_image[u,v] = index

        return depth_image, index_image

    def generate_grasp(self, grasp_axis, grasp_center):
        """
        `Input`
        array 3 :grasp_axis
        array 3 :grasp_center \n
        `return`
        list: wrapped_surfs (in obj fram)
        float: grasp_depth
        """
        grasp_depth = 0
        [max_depth, radius] = self.gripper_config_
        while grasp_depth < max_depth:
            wrapped_surfs =  []
            for x_surf in self.surfpoints_in_obj_:
                if abs((x_surf-grasp_center).dot(grasp_axis) - grasp_depth) < 0.0001:
                    # check collision
                    if np.linalg.norm(x_surf - (grasp_center+grasp_axis*grasp_depth)) > radius:
                        return wrapped_surfs, grasp_depth
                elif (x_surf-grasp_center).dot(grasp_axis) < grasp_depth:
                    wrapped_surfs.append(x_surf)
            grasp_depth += max_depth/20
        return wrapped_surfs, grasp_depth
        
    def force_closure(self,wrapped_surfs, grasp_axis):
        """
        `return`
        float : normal/max_friction(mu=1) 
        """
        grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)

        if len(wrapped_surfs) == 0:
            return -1
        else:
            # trasfer surf into Class Contact
            contacts = []
            for surf in wrapped_surfs:
                contact = Contact3D(self.obj_, surf)
                contacts.append(contact)
            
            total_axis_friction = 0
            total_axis_normal = 0
            for k, contact in enumerate(contacts):
                if contact is None or contact.point_ is None or contact.normal_ is None:
                    continue
                normal = contact.normal_
                normal = normal / np.linalg.norm(normal)
                axis_normal = grasp_axis.dot(normal) # normal force alone grasp_axis
                total_axis_normal += axis_normal
                theta = np.arccos(axis_normal)
                total_axis_friction += abs(np.linalg.norm(normal)* 1 * np.sin(theta))
            return total_axis_friction /abs(total_axis_normal)

    def generate_quality(self):
        T_camera_in_obj = self.T_table_in_obj_.dot(self.T_camera_in_table_)
        grasp_axis_in_obj = T_camera_in_obj[:3, 2] #- self.T_camera_in_table_[:3, 2] # verical grasp
        quality_img = np.zeros(self.picture_size_) - 1
        max_quality = 0
        for i in range(self.picture_size_[0]):
            for j in range(self.picture_size_[1]):
                index = self.index_img_[i,j]
                # print("loading----",100*(i*self.picture_size_[1]+j)/(self.picture_size_[1]*self.picture_size_[0]),"%")
                if index != -1:
                    # print("11111")
                    grasp_center_in_obj = self.surfpoints_in_obj_[index]
                    wrapped_surfs, grasp_depth = self.generate_grasp(grasp_axis_in_obj, grasp_center_in_obj)
                    num_wrapped_surfs = len(wrapped_surfs)
                    # friction coefficience
                    mu = self.force_closure(wrapped_surfs, grasp_axis_in_obj)
                    # grasp wrench space quality
                    
                    if mu > 0:
                        forces = np.zeros([3, 0])
                        torques = np.zeros([3, 0])
                        normals = np.zeros([3, 0])
                        for surf_index in range(num_wrapped_surfs):
                            contact = Contact3D(self.obj_, wrapped_surfs[surf_index])
                            # get contact forces
                            force_success, contact_forces, contact_outward_normal = contact.friction_cone(8, mu)
                            if not force_success:
                                print('Force computation failed')
                                break
                            torque_success, contact_torques = contact.torques(contact_forces)
                            if not torque_success:
                                print('Torque computation failed')
                                break
                            forces = np.c_[forces, contact_forces]
                            torques = np.c_[torques, contact_torques]
                            normals = np.c_[normals, -contact_outward_normal]  # store inward pointing normals
                        canny_quality = PointGraspMetrics3D.ferrari_canny_L1_force_only(forces, torques, normals)
                        print("canny_quality:",canny_quality," fc_quality:",1/mu)
                        quality =  (1/mu) + 0.1 * canny_quality # grasp_depth/self.gripper_config_[0] *
                        print("quality:",quality)
                        if quality > max_quality: max_quality = quality
                    else:
                        quality = 0
                else:
                    quality = 0
                
                quality_img[i,j] = quality
        return quality_img, max_quality

def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count("/") == file_dir_.count("/") + 1:
            file_list.append(root)
    file_list.sort()
    return file_list

if __name__ == "__main__":
    camera_matrix = [500, 4, 4, 400/2, 400/2]
    gripper_config = [0.2, 0.03]
    pointnetgpd_dir = os.environ["PointNetGPD_FOLDER"]
    file_dir = pointnetgpd_dir + "/PointNetGPD/data/ycb-tools/models/ycb"
    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()
    print("object_numbers:",object_numbers)
    for i in range(object_numbers):
        object_name = file_list_all[i].split("/")[-1]
        print("dealing---",object_name)
        if os.path.exists(str(file_list_all[i]) + "/google_512k/nontextured.obj"):
            of = ObjFile(str(file_list_all[i]) + "/google_512k/nontextured.obj")
            sf = SdfFile(str(file_list_all[i]) + "/google_512k/nontextured.sdf")
        else:
            print("can not find any obj or sdf file!")
            continue
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)
        print("Log: opened object", i + 1, object_name)

        for k in range(10):
            heat_map = Heatmap_Generate(obj, camera_matrix, gripper_config)
            # plt.imshow(heat_map.quality_img_) 
            np.save("generatenpy/quality_" + str(k) + ".npy", heat_map.quality_img_)
            np.save("generatenpy/depth_" + str(k) + ".npy", heat_map.depth_img_)


