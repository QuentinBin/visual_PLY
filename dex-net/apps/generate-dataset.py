import numpy as np
import sys
import pickle
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler, CylinderGraspSampler
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os
import multiprocessing
import logging

def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count("/") == file_dir_.count("/") + 1:
            file_list.append(root)
    file_list.sort()
    return file_list

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename_prefix = sys.argv[1]
    else:
        filename_prefix = "default"
    pointnetgpd_dir = os.environ["PointNetGPD_FOLDER"]
    file_dir = pointnetgpd_dir + "/PointNetGPD/data/ycb-tools/models/ycb"
    yaml_config = YamlConfig(pointnetgpd_dir + "/dex-net/test/config.yaml")
    gripper_name = "robotiq_85"
    gripper = RobotGripper.load(gripper_name, pointnetgpd_dir + "/dex-net/data/grippers")
    grasp_sample_method = "usr"
    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()
    print(object_numbers)
    print("-------------------------------------------------")
    for i in range(object_numbers):
        object_name = file_list_all[i][len(os.environ["HOME"]) + 35:]
        print("a worker of task {} start".format(object_name))
        ags = CylinderGraspSampler(gripper, yaml_config)
        print("Log: do job", i)
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

        force_closure_quality_config = {}
        canny_quality_config = {}
        fc_list = np.arange(0.6, 0.2, -0.1)#, 0.6  ##################################################
        for value_fc in fc_list:
            value_fc = round(value_fc, 2)
            yaml_config["metrics"]["force_closure"]["friction_coef"] = value_fc
            yaml_config["metrics"]["robust_ferrari_canny"]["friction_coef"] = value_fc
            print("yaml config force_closure:",yaml_config["metrics"]["force_closure"])
            force_closure_quality_config[value_fc] =  GraspQualityConfigFactory.create_config(yaml_config["metrics"]["force_closure"])
            canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(yaml_config["metrics"]["robust_ferrari_canny"])
        good_count_perfect = np.zeros(len(fc_list))
        count = 0
        minimum_grasp_per_fc = 1 ##############################################
        sample_nums = 100 #############################################
        good_grasp = []
        # while np.sum(good_count_perfect < minimum_grasp_per_fc) != 0:
        grasps,frictions = ags.generate_grasps(obj, target_num_grasps=sample_nums, grasp_gen_mult=10,
                                    vis=False, random_approach_angle=True)
        for inde, gra in enumerate(grasps):
            good_grasp.append((gra, frictions[inde]))
            # count += len(grasps)
            # for inde, j in enumerate(grasps):
            #     print("----------qualitifying grasp",inde,"-------------")
            #     tmp, is_force_closure = False, False
            #     for ind_, value_fc in enumerate(fc_list):
            #         tmp = is_force_closure
            #         value_fc = round(value_fc, 2)
            #         is_force_closure = PointGraspMetrics3D.grasp_quality(j, obj,
            #                                                             force_closure_quality_config[round(value_fc, 2)], vis=False)
            #         if tmp and not is_force_closure:
            #             if good_count_perfect[ind_ - 1] < minimum_grasp_per_fc:
            #                 canny_quality = PointGraspMetrics3D.grasp_quality(j, obj,
            #                                                                 canny_quality_config[round(fc_list[ind_ - 1], 2)],
            #                                                                 vis=False)
            #                 good_grasp.append((j, fc_list[ind_ - 1], canny_quality))
            #                 good_count_perfect[ind_ - 1] += 1
            #             break
            #         elif is_force_closure and value_fc == fc_list[-1]:
            #             if good_count_perfect[ind_] < minimum_grasp_per_fc:
            #                 canny_quality = PointGraspMetrics3D.grasp_quality(j, obj,
            #                                                                 canny_quality_config[round(value_fc, 2)], vis=False)
            #                 good_grasp.append((j, value_fc, canny_quality))
            #                 good_count_perfect[ind_] += 1
            #             break
            # print("Object:{} GoodGrasp:{}".format(object_name, good_count_perfect))
        object_name_len = len(object_name)
        object_name_ = str(object_name) + " " * (25 - object_name_len)
        if count == 0:
            good_grasp_rate = 0
        else:
            good_grasp_rate = len(good_grasp) / count
        print("Gripper:{} Object:{} Rate:{:.4f} {}/{}".
            format(gripper_name, object_name_, good_grasp_rate, len(good_grasp), count))

        # save
        good_grasp_file_name = "./generated_grasps/{}_{}_{}".format(filename_prefix,object_name_.split("/")[-1], str(len(good_grasp)))
        with open(good_grasp_file_name + ".pickle", "wb") as f:
            pickle.dump(good_grasp, f)

        tmp = []
        for grasp in good_grasp:
            grasp_config = grasp[0].configuration
            score_friction = grasp[1]
            # score_canny = grasp[2]
            tmp.append(np.concatenate([grasp_config, [score_friction]]))
        print(tmp)
        np.save(good_grasp_file_name + ".npy", np.array(tmp))
        print("finished job ", object_name)
    print("All job done.")
