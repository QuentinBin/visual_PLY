# -*- coding: utf-8 -*-
# """
# Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational,
# research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
# hereby granted, provided that the above copyright notice, this paragraph and the following two
# paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
# Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94360-1620, (510) 643-
# 3601, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
# THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# """
# """
# Grasp class that implements gripper endpoints and grasp functions
# Authors: Jeff Mahler, with contributions from Jacky Liang and Nikhil Sharma
# """
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm
import scipy.linalg as linalg
import time

from autolab_core import Point, RigidTransform
from meshpy import Sdf3D, StablePose

try:
    from gqcnn import Grasp2D
except:
    pass
    # logging.warning('Failed to import gqcnn! Grasp2D functions will not be available.')

from dexnet import abstractstatic
from dexnet.grasping import Contact3D, GraspableObject3D


# class Grasp(object, metaclass=ABCMeta):
class Grasp(object):
    """ Abstract grasp class.

    Attributes
    ----------
    configuration : :obj:`numpy.ndarray`
        vector specifying the parameters of the grasp (e.g. hand pose, opening width, joint angles, etc)
    frame : :obj:`str`
        string name of grasp reference frame (defaults to obj)
    """
    __metaclass__ = ABCMeta
    samples_per_grid = 2  # global resolution for line of action

    @abstractmethod
    def close_fingers(self, obj):
        """ Finds the contact points by closing on the given object.
        
        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to find contacts on
        """
        pass

    @abstractmethod
    def configuration(self):
        """ Returns the numpy array representing the hand configuration """
        pass

    @abstractmethod
    def frame(self):
        """ Returns the string name of the grasp reference frame  """
        pass

    @abstractstatic
    def params_from_configuration(configuration):
        """ Convert configuration vector to a set of params for the class """
        pass

    @abstractstatic
    def configuration_from_params(*params):
        """ Convert param list to a configuration vector for the class """
        pass


# class PointGrasp(Grasp, metaclass=ABCMeta):
class PointGrasp(Grasp):
    """ Abstract grasp class for grasps with a point contact model.

    Attributes
    ----------
    configuration : :obj:`numpy.ndarray`
        vector specifying the parameters of the grasp (e.g. hand pose, opening width, joint angles, etc)
    frame : :obj:`str`
        string name of grasp reference frame (defaults to obj)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_line_of_action(g, axis, width, obj, num_samples):
        """ Creates a line of action, or the points in space that the grasp traces out, from a point g in world coordinates on an object.

        Returns
        -------
        bool
            whether or not successful
        :obj:`list` of :obj:`numpy.ndarray`
            points in 3D space along the line of action
        """
        pass

    @abstractmethod
    def create_cylinder_of_action(center, axis, depth, radius, obj, num_samples):
        """创建圆柱收缩的路径点"""
        pass

    # NOTE: implementation of close_fingers must return success, array of contacts (one per column)


class ParallelJawPtGrasp3D(PointGrasp):
    """ Parallel Jaw point grasps in 3D space.
    """

    def __init__(self, configuration, frame='object', grasp_id=None):
        # get parameters from configuration array
        grasp_center, grasp_axis, grasp_width, grasp_angle, jaw_width, min_grasp_width = \
            ParallelJawPtGrasp3D.params_from_configuration(configuration)

        self.center_ = grasp_center
        self.axis_ = grasp_axis / np.linalg.norm(grasp_axis)
        self.max_grasp_width_ = grasp_width
        self.jaw_width_ = jaw_width
        self.min_grasp_width_ = min_grasp_width
        self.approach_angle_ = grasp_angle
        self.frame_ = frame
        self.grasp_id_ = grasp_id

    @property
    def center(self):
        """ :obj:`numpy.ndarray` : 3-vector specifying the center of the jaws """
        return self.center_

    @center.setter
    def center(self, x):
        self.center_ = x

    @property
    def axis(self):
        """ :obj:`numpy.ndarray` : normalized 3-vector specifying the line between the jaws """
        return self.axis_

    @property
    def open_width(self):
        """ float : maximum opening width of the jaws """
        return self.max_grasp_width_

    @property
    def close_width(self):
        """ float : minimum opening width of the jaws """
        return self.min_grasp_width_

    @property
    def jaw_width(self):
        """ float : width of the jaws in the tangent plane to the grasp axis """
        return self.jaw_width_

    @property
    def approach_angle(self):
        """ float : approach angle of the grasp """
        return self.approach_angle_

    @property
    def configuration(self):
        """ :obj:`numpy.ndarray` : vector specifying the parameters of the grasp as follows
        (grasp_center, grasp_axis, grasp_angle, grasp_width, jaw_width) """
        return ParallelJawPtGrasp3D.configuration_from_params(self.center_, self.axis_, self.max_grasp_width_,
                                                              self.approach_angle_, self.jaw_width_,
                                                              self.min_grasp_width_)

    @property
    def frame(self):
        """ :obj:`str` : name of grasp reference frame """
        return self.frame_

    @property
    def id(self):
        """ int : id of grasp """
        return self.grasp_id_

    @frame.setter
    def frame(self, f):
        self.frame_ = f

    @approach_angle.setter
    def approach_angle(self, angle):
        """ Set the grasp approach angle """
        self.approach_angle_ = angle

    @property
    def endpoints(self):
        """
        Returns
        -------
        :obj:`numpy.ndarray`
            location of jaws in 3D space at max opening width """
        return self.center_ - (self.max_grasp_width_ / 2.0) * self.axis_, self.center_ + (
                self.max_grasp_width_ / 2.0) * self.axis_,

    @staticmethod
    def distance(g1, g2, alpha=0.05):
        """ Evaluates the distance between two grasps.

        Parameters
        ----------
        g1 : :obj:`ParallelJawPtGrasp3D`
            the first grasp to use
        g2 : :obj:`ParallelJawPtGrasp3D`
            the second grasp to use
        alpha : float
            parameter weighting rotational versus spatial distance

        Returns
        -------
        float
            distance between grasps g1 and g2
        """
        center_dist = np.linalg.norm(g1.center - g2.center)
        axis_dist = (2.0 / np.pi) * np.arccos(np.abs(g1.axis.dot(g2.axis)))
        return center_dist + alpha * axis_dist

    @staticmethod
    def configuration_from_params(center, axis, width, angle=0, jaw_width=0, min_width=0):
        """ Converts grasp parameters to a configuration vector. """
        if np.abs(np.linalg.norm(axis) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        configuration = np.zeros(10)
        configuration[0:3] = center
        configuration[3:6] = axis
        configuration[6] = width
        configuration[7] = angle
        configuration[8] = jaw_width
        configuration[9] = min_width
        return configuration

    @staticmethod
    def params_from_configuration(configuration):
        """ Converts configuration vector into grasp parameters.
        
        Returns
        -------
        grasp_center : :obj:`numpy.ndarray`
            center of grasp in 3D space
        grasp_axis : :obj:`numpy.ndarray`
            normalized axis of grasp in 3D space
        max_width : float
            maximum opening width of jaws
        angle : float
            approach angle
        jaw_width : float
            width of jaws
        min_width : float
            minimum closing width of jaws
        """
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 10):
            raise ValueError('Configuration must be numpy ndarray of size 9 or 10')
        if configuration.shape[0] == 9:
            min_grasp_width = 0
        else:
            min_grasp_width = configuration[9]
        if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        return configuration[0:3], configuration[3:6], configuration[6], configuration[7], configuration[
            8], min_grasp_width

    @staticmethod
    def center_from_endpoints(g1, g2):
        """ Grasp center from endpoints as np 3-arrays """
        grasp_center = (g1 + g2) / 2
        return grasp_center

    @staticmethod
    def axis_from_endpoints(g1, g2):
        """ Normalized axis of grasp from endpoints as np 3-arrays """
        grasp_axis = g2 - g1
        if np.linalg.norm(grasp_axis) == 0:
            return grasp_axis
        return grasp_axis / np.linalg.norm(grasp_axis)

    @staticmethod
    def width_from_endpoints(g1, g2):
        """ Width of grasp from endpoints """
        grasp_axis = g2 - g1
        return np.linalg.norm(grasp_axis)

    @staticmethod
    def grasp_from_endpoints(g1, g2, width=None, approach_angle=0, close_width=0):
        """ Create a grasp from given endpoints in 3D space, making the axis the line between the points.

        Parameters
        ---------
        g1 : :obj:`numpy.ndarray`
            location of the first jaw
        g2 : :obj:`numpy.ndarray`
            location of the second jaw
        width : float
            maximum opening width of jaws
        approach_angle : float
            approach angle of grasp
        close_width : float
            width of gripper when fully closed
        """
        x = ParallelJawPtGrasp3D.center_from_endpoints(g1, g2)
        v = ParallelJawPtGrasp3D.axis_from_endpoints(g1, g2)
        if width is None:
            width = ParallelJawPtGrasp3D.width_from_endpoints(g1, g2)
        return ParallelJawPtGrasp3D(
            ParallelJawPtGrasp3D.configuration_from_params(x, v, width, min_width=close_width, angle=approach_angle))

    @property
    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the 0-degree approach direction, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
        if np.linalg.norm(grasp_axis_x) == 0:
            grasp_axis_x = np.array([1, 0, 0])
        grasp_axis_x = grasp_axis_x / norm(grasp_axis_x)
        grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        return R

    @property
    def rotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the grasp approach angle, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        R = ParallelJawPtGrasp3D._get_rotation_matrix_y(self.approach_angle)
        R = self.unrotated_full_axis.dot(R)
        return R

    @property
    def T_grasp_obj(self):
        """ Rigid transformation from grasp frame to object frame.
        Rotation matrix is X-axis along approach direction, Y axis pointing between the jaws, and Z-axis orthogonal.
        Translation vector is the grasp center.

        Returns
        -------
        :obj:`RigidTransform`
            transformation from grasp to object coordinates
        """
        T_grasp_obj = RigidTransform(self.rotated_full_axis, self.center, from_frame='grasp', to_frame='obj')
        return T_grasp_obj

    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R

    def gripper_pose(self, gripper=None):
        """ Returns the RigidTransformation from the gripper frame to the object frame when the gripper is executing the
        given grasp.
        Differs from the grasp reference frame because different robots use different conventions for the gripper
        reference frame.
        
        Parameters
        ----------
        gripper : :obj:`RobotGripper`
            gripper to get the pose for

        Returns
        -------
        :obj:`RigidTransform`
            transformation from gripper frame to object frame
        """
        if gripper is None:
            T_gripper_grasp = RigidTransform(from_frame='gripper', to_frame='grasp')
        else:
            T_gripper_grasp = gripper.T_grasp_gripper

        T_gripper_obj = self.T_grasp_obj * T_gripper_grasp
        return T_gripper_obj

    def grasp_angles_from_stp_z(self, stable_pose):
        """ Get angles of the the grasp from the table plane:
        1) the angle between the grasp axis and table normal
        2) the angle between the grasp approach axis and the table normal
        
        Parameters
        ----------
        stable_pose : :obj:`StablePose` or :obj:`RigidTransform`
            the stable pose to compute the angles for

        Returns
        -------
        psi : float
            grasp y axis rotation from z axis in stable pose
        phi : float
            grasp x axis rotation from z axis in stable pose
        """
        T_grasp_obj = self.T_grasp_obj

        if isinstance(stable_pose, StablePose):
            R_stp_obj = stable_pose.r
        else:
            R_stp_obj = stable_pose.rotation
        T_stp_obj = RigidTransform(R_stp_obj, from_frame='obj', to_frame='stp')

        T_stp_grasp = T_stp_obj * T_grasp_obj

        stp_z = np.array([0, 0, 1])
        grasp_axis_angle = np.arccos(stp_z.dot(T_stp_grasp.y_axis))
        grasp_approach_angle = np.arccos(abs(stp_z.dot(T_stp_grasp.x_axis)))
        nu = stp_z.dot(T_stp_grasp.z_axis)

        return grasp_axis_angle, grasp_approach_angle, nu

    def close_fingers(self, obj, vis=False, check_approach=True, approach_dist=1.0):
        """ Steps along grasp axis to find the locations of contact with an object

        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to close fingers on
        vis : bool
            whether or not to plot the line of action and contact points
        check_approach : bool
            whether or not to check if the contact points can be reached
        approach_dist : float
            how far back to check the approach distance, only if checking the approach is set
        
        Returns
        -------
        success : bool
            whether or not contacts were found
        c1 : :obj:`Contact3D`
            the contact point for jaw 1
        c2 : :obj:`Contact3D`
            the contact point for jaw 2
        """
        if vis:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()
        # compute num samples to use based on sdf resolution
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(self.max_grasp_width_)
        num_samples = int(Grasp.samples_per_grid * float(grasp_width_grid) / 2)  # at least 1 sample per grid

        # get grasp endpoints in sdf frame
        g1_world, g2_world = self.endpoints

        # check for contact along approach
        if check_approach:
            approach_dist_grid = obj.sdf.transform_pt_obj_to_grid(approach_dist)
            num_approach_samples = int(Grasp.samples_per_grid * approach_dist_grid / 2)  # at least 1 sample per grid
            approach_axis = self.rotated_full_axis[:, 0]
            approach_loa1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, -approach_axis, approach_dist, obj,
                                                                       num_approach_samples, min_width=0)
            approach_loa2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -approach_axis, approach_dist, obj,
                                                                       num_approach_samples, min_width=0)
            c1_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa1, obj, vis=vis, strict=True)
            c2_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa2, obj, vis=vis, strict=True)
            approach_collision = c1_found or c2_found
            if approach_collision:
                plt.clf()
                return False, None

        # get line of action            
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)

        if vis:
            ax = plt.gca(projection='3d')
            surface = obj.sdf.surface_points()[0]
            surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
            ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                       s=np.ones_like(surface[:, 0]) * 0.3, c='b')

        # find contacts
        c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
        c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)

        if vis:
            ax = plt.gca(projection='3d')
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()

        contacts_found = c1_found and c2_found
        return contacts_found, [c1, c2]

    def vis_grasp(self, obj, *args, **kwargs):
        if 'keep' not in kwargs or not kwargs['keep']:
            plt.clf()

        ax = plt.gca(projection='3d')
        if 'show_obj' in kwargs and kwargs['show_obj']:
            # plot the obj
            surface = obj.sdf.surface_points()[0]
            surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
            ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                       s=np.ones_like(surface[:, 0]) * 0.3, c='b')

        # plot the center of grasp using grid
        grasp_center_grid = obj.sdf.transform_pt_obj_to_grid(self.center)
        ax.scatter(grasp_center_grid[0], grasp_center_grid[1], grasp_center_grid[2], marker='x', c='r')

        # compute num samples to use based on sdf resolution
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(self.max_grasp_width_)
        num_samples = int(Grasp.samples_per_grid * float(grasp_width_grid) / 2)  # at least 1 sample per grid

        # get grasp endpoints in sdf frame
        g1_world, g2_world = self.endpoints

        # check for contact along approach
        approach_dist = 0.1
        approach_dist_grid = obj.sdf.transform_pt_obj_to_grid(approach_dist)
        num_approach_samples = int(approach_dist_grid / 2)  # at least 1 sample per grid
        approach_axis = self.rotated_full_axis[:, 0]
        approach_loa1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, -approach_axis, approach_dist, obj,
                                                                   num_approach_samples, min_width=0)
        approach_loa2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -approach_axis, approach_dist, obj,
                                                                   num_approach_samples, min_width=0)
        end1, end2 = approach_loa1[-1], approach_loa2[-1]
        begin1, begin2 = approach_loa1[0], approach_loa2[0]
        ax.plot([end1[0], end2[0]], [end1[1], end2[1]], [end1[2], end2[2]], 'r-', linewidth=5)
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], 'r-', linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], 'r-', linewidth=5)
        c1_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa1, obj, vis=False, strict=True)
        c2_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa2, obj, vis=False, strict=True)
        approach_collision = c1_found or c2_found
        if approach_collision:
            plt.clf()
            return False

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)

        # find contacts
        c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=False)
        c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=False)
        begin1, begin2 = line_of_action1[0], line_of_action2[0]
        end1, end2 = obj.sdf.transform_pt_obj_to_grid(c1.point), obj.sdf.transform_pt_obj_to_grid(c2.point)
        print(end1, end2)
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], 'r-', linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], 'r-', linewidth=5)
        ax.scatter(end1[0], end1[1], end1[2], s=80, c='g')
        ax.scatter(end2[0], end2[1], end2[2], s=80, c='g')

        ax.set_xlim3d(0, obj.sdf.dims_[0])
        ax.set_ylim3d(0, obj.sdf.dims_[1])
        ax.set_zlim3d(0, obj.sdf.dims_[2])
        plt.title(','.join([str(i) for i in args]))
        plt.draw()

        contacts_found = c1_found and c2_found
        return contacts_found

    @staticmethod
    def create_line_of_action(g, axis, width, obj, num_samples, min_width=0, convert_grid=True):
        """
        Creates a straight line of action, or list of grid points, from a given point and direction in world or grid coords

        Parameters
        ----------
        g : 3x1 :obj:`numpy.ndarray`
            start point to create the line of action
        axis : normalized 3x1 :obj:`numpy.ndarray`
            normalized numpy 3 array of grasp direction
        width : float
            the grasp width
        num_samples : int
            number of discrete points along the line of action
        convert_grid : bool
            whether or not the points are specified in world coords

        Returns
        -------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarrays`
            coordinates to pass through in 3D space for contact checking
        """
        num_samples = max(num_samples, 3)  # always at least 3 samples
        line_of_action = [g + t * axis for t in
                          np.linspace(0, float(width) / 2 - float(min_width) / 2, num=num_samples)]
        if convert_grid:
            as_array = np.array(line_of_action).T
            transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
            line_of_action = list(transformed.T)
        return line_of_action

    @staticmethod
    def find_contact(line_of_action, obj, vis=False, strict=False):
        """
        Find the point at which a point traveling along a given line of action hits a surface.

        Parameters
        ----------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            the points visited as the fingers close (grid coords)
        obj : :obj:`GraspableObject3D`
            to check contacts on
        vis : bool
            whether or not to display the contact check (for debugging)

        Returns
        -------
        contact_found : bool
            whether or not the point contacts the object surface
        contact : :obj:`Contact3D`
            found along line of action (None if contact not found)
        """
        contact_found = False
        pt_zc = None
        pt_zc_world = None
        contact = None
        num_pts = len(line_of_action)
        sdf_here = 0
        sdf_before = 0
        pt_grid = None
        pt_before = None

        # step along line of action, get points on surface when possible
        i = 0
        while i < num_pts and not contact_found:
            # update loop vars
            pt_before_before = pt_before
            pt_before = pt_grid
            sdf_before_before = sdf_before
            sdf_before = sdf_here
            pt_grid = line_of_action[i]

            # visualize
            if vis:
                ax = plt.gca(projection='3d')
                ax.scatter(pt_grid[0], pt_grid[1], pt_grid[2], c='r')

            # check surface point
            on_surface, sdf_here = obj.sdf.on_surface(pt_grid)
            if on_surface:
                contact_found = True
                if strict:
                    return contact_found, None

                # quadratic approximation to find actual zero crossing
                if i == 0:
                    pt_after = line_of_action[i + 1]
                    sdf_after = obj.sdf[pt_after]
                    pt_after_after = line_of_action[i + 2]
                    sdf_after_after = obj.sdf[pt_after_after]

                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_grid, sdf_here, pt_after, sdf_after, pt_after_after,
                                                               sdf_after_after)

                    # contact not yet found if next sdf value is smaller
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False

                elif i == len(line_of_action) - 1:
                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_before_before, sdf_before_before, pt_before,
                                                               sdf_before, pt_grid, sdf_here)

                    if pt_zc is None:
                        contact_found = False

                else:
                    pt_after = line_of_action[i + 1]
                    sdf_after = obj.sdf[pt_after]
                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_before, sdf_before, pt_grid, sdf_here, pt_after,
                                                               sdf_after)

                    # contact not yet found if next sdf value is smaller
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False
            i = i + 1

        # visualization
        if vis and contact_found:
            ax = plt.gca(projection='3d')
            ax.scatter(pt_zc[0], pt_zc[1], pt_zc[2], s=80, c='g')

        if contact_found:
            pt_zc_world = obj.sdf.transform_pt_grid_to_obj(pt_zc)
            in_direction_grid = line_of_action[-1] - line_of_action[0]
            in_direction_grid = in_direction_grid / np.linalg.norm(in_direction_grid)
            in_direction = obj.sdf.transform_pt_grid_to_obj(in_direction_grid, direction=True)
            contact = Contact3D(obj, pt_zc_world, in_direction=in_direction)
            if contact.normal is None:
                contact_found = False
        return contact_found, contact

    def _angle_aligned_with_stable_pose(self, stable_pose):
        """
        Returns the y-axis rotation angle that'd allow the current pose to align with stable pose.
        """

        def _argmin(f, a, b, n):
            # finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            min_y = f(a)
            min_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y <= min_y:
                    min_y = y
                    min_x = x
            return min_x

        def _get_matrix_product_x_axis(grasp_axis, normal):
            def matrix_product(theta):
                R = ParallelJawPtGrasp3D._get_rotation_matrix_y(theta)
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return abs(np.dot(normal, grasp_axis_rotated))

            return matrix_product

        stable_pose_normal = stable_pose.r[2, :]

        theta = _argmin(
            _get_matrix_product_x_axis(np.array([1, 0, 0]), np.dot(inv(self.unrotated_full_axis), stable_pose_normal)),
            0, 2 * np.pi, 1000)
        return theta

    def grasp_y_axis_offset(self, theta):
        """ Return a new grasp with the given approach angle.
        
        Parameters
        ----------
        theta : float
            approach angle for the new grasp

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            grasp with the given approach angle
        """
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta + self.approach_angle
        return new_grasp

    def parallel_table(self, stable_pose):
        """
        Returns a grasp with approach_angle set to be perpendicular to the table normal specified in the given stable pose.

        Parameters
        ----------
        stable_pose : :obj:`StablePose`
            the pose specifying the table

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            aligned grasp
        """
        theta = self._angle_aligned_with_stable_pose(stable_pose)
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta
        return new_grasp

    def _angle_aligned_with_table(self, table_normal):
        """
        Returns the y-axis rotation angle that'd allow the current pose to align with the table normal.
        """

        def _argmax(f, a, b, n):
            # finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            max_y = f(a)
            max_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y >= max_y:
                    max_y = y
                    max_x = x
            return max_x

        def _get_matrix_product_x_axis(grasp_axis, normal):
            def matrix_product(theta):
                R = ParallelJawPtGrasp3D._get_rotation_matrix_y(theta)
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return np.dot(normal, grasp_axis_rotated)

            return matrix_product

        theta = _argmax(
            _get_matrix_product_x_axis(np.array([1, 0, 0]), np.dot(inv(self.unrotated_full_axis), -table_normal)), 0,
            2 * np.pi, 64)
        return theta

    def perpendicular_table(self, stable_pose):
        """
        Returns a grasp with approach_angle set to be aligned width the table normal specified in the given stable pose.

        Parameters
        ----------
        stable_pose : :obj:`StablePose` or :obj:`RigidTransform`
            the pose specifying the orientation of the table

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            aligned grasp
        """
        if isinstance(stable_pose, StablePose):
            table_normal = stable_pose.r[2, :]
        else:
            table_normal = stable_pose.rotation[2, :]
        theta = self._angle_aligned_with_table(table_normal)
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta
        return new_grasp

    def project_camera(self, T_obj_camera, camera_intr):
        """ Project a grasp for a given gripper into the camera specified by a set of intrinsics.
        
        Parameters
        ----------
        T_obj_camera : :obj:`autolab_core.RigidTransform`
            rigid transformation from the object frame to the camera frame
        camera_intr : :obj:`perception.CameraIntrinsics`
            intrinsics of the camera to use
        """
        # compute pose of grasp in camera frame
        T_grasp_camera = T_obj_camera * self.T_grasp_obj
        y_axis_camera = T_grasp_camera.y_axis[:2]
        if np.linalg.norm(y_axis_camera) > 0:
            y_axis_camera = y_axis_camera / np.linalg.norm(y_axis_camera)

        # compute grasp axis rotation in image space
        rot_z = np.arccos(y_axis_camera[0])
        if y_axis_camera[1] < 0:
            rot_z = -rot_z
        while rot_z < 0:
            rot_z += 2 * np.pi
        while rot_z > 2 * np.pi:
            rot_z -= 2 * np.pi

        # compute grasp center in image space
        t_grasp_camera = T_grasp_camera.translation
        p_grasp_camera = Point(t_grasp_camera, frame=camera_intr.frame)
        u_grasp_camera = camera_intr.project(p_grasp_camera)
        d_grasp_camera = t_grasp_camera[2]
        return Grasp2D(u_grasp_camera, rot_z, d_grasp_camera,
                       width=self.open_width,
                       camera_intr=camera_intr)

    @staticmethod
    def grasp_from_contact_and_axis_on_grid(obj, grasp_c1_world, grasp_axis_world, grasp_width_world, grasp_angle=0,
                                            jaw_width_world=0,
                                            min_grasp_width_world=0, vis=False, backup=0.5):
        """
        Creates a grasp from a single contact point in grid coordinates and direction in grid coordinates.
        
        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to create grasp for
        grasp_c1_grid : 3x1 :obj:`numpy.ndarray`
            contact point 1 in world
        grasp_axis : normalized 3x1 :obj:`numpy.ndarray`
           normalized direction of the grasp in world
        grasp_width_world : float
            grasp_width in world coords
        jaw_width_world : float
            width of jaws in world coords
        min_grasp_width_world : float
            min closing width of jaws
        vis : bool
            whether or not to visualize the grasp
        
        Returns
        -------
        g : :obj:`ParallelJawGrasp3D`
            grasp created by finding the second contact
        c1 : :obj:`Contact3D`
            first contact point on the object
        c2 : :obj:`Contact3D`
            second contact point on the object
        """
        # transform to grid basis
        grasp_axis_world = grasp_axis_world / np.linalg.norm(grasp_axis_world)
        grasp_axis_grid = obj.sdf.transform_pt_obj_to_grid(grasp_axis_world, direction=True)
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(grasp_width_world)
        min_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(min_grasp_width_world)
        grasp_c1_grid = obj.sdf.transform_pt_obj_to_grid(
            grasp_c1_world) - backup * grasp_axis_grid  # subtract to find true point
        num_samples = int(2 * grasp_width_grid)  # at least 2 samples per grid
        g2 = grasp_c1_grid + (grasp_width_grid - backup) * grasp_axis_grid

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(grasp_c1_grid, grasp_axis_grid, grasp_width_grid,
                                                                     obj, num_samples,
                                                                     min_width=min_grasp_width_grid, convert_grid=False)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2, -grasp_axis_grid, 2 * grasp_width_grid, obj,
                                                                     num_samples,
                                                                     min_width=0, convert_grid=False)
        if vis:
            obj.sdf.scatter()
            ax = plt.gca(projection='3d')
            ax.scatter(grasp_c1_grid[0] - grasp_axis_grid[0], grasp_c1_grid[1] - grasp_axis_grid[1],
                       grasp_c1_grid[2] - grasp_axis_grid[2], c='r')
            ax.scatter(grasp_c1_grid[0], grasp_c1_grid[1], grasp_c1_grid[2], s=80, c='b')

        # compute the contact points on the object
        contact1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
        contact2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)

        if vis:
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()
        if not contact1_found or not contact2_found or np.linalg.norm(c1.point - c2.point) <= min_grasp_width_world:
            logging.debug('No contacts found for grasp')
            return None, None, None

        # create grasp
        grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c1.point, c2.point)
        grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c1.point, c2.point)
        configuration = ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, grasp_width_world,
                                                                       grasp_angle, jaw_width_world)
        return ParallelJawPtGrasp3D(configuration), c1, c2  # relative to object

    def surface_information(self, graspable, width=2e-2, num_steps=21, direction=None):
        """ Return the patch surface information at the contacts that this grasp makes on a graspable.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            object to get surface information for
        width : float
            width of the window in obj frame
        num_steps : int
            number of steps

        Returns
        -------
        :obj:`list` of :obj:`SurfaceWindow`
             surface patches, one for each contact
        """
        return graspable.surface_information(self, width, num_steps, direction1=self.axis_, direction2=-self.axis_)


class VacuumPoint(Grasp):
    """ Defines a vacuum target point and axis in 3D space (5 DOF)
    """

    def __init__(self, configuration, frame='object', grasp_id=None):
        center, axis = VacuumPoint.params_from_configuration(configuration)
        self._center = center
        self._axis = axis
        self.frame_ = frame

    @property
    def center(self):
        return self._center

    @property
    def axis(self):
        return self._axis

    @property
    def frame(self):
        return self._frame

    @property
    def configuration(self):
        return VacuumPoint.configuration_from_params(self._center, self._axis)

    @staticmethod
    def configuration_from_params(center, axis):
        """ Converts grasp parameters to a configuration vector. """
        if np.abs(np.linalg.norm(axis) - 1.0) > 1e-5:
            raise ValueError('Illegal vacuum axis. Must be norm one')
        configuration = np.zeros(6)
        configuration[0:3] = center
        configuration[3:6] = axis
        return configuration

    @staticmethod
    def params_from_configuration(configuration):
        """ Converts configuration vector into vacuum grasp parameters.
        
        Returns
        -------
        center : :obj:`numpy.ndarray`
            center of grasp in 3D space
        axis : :obj:`numpy.ndarray`
            normalized axis of grasp in 3D space
        """
        if not isinstance(configuration, np.ndarray) or configuration.shape[0] != 6:
            raise ValueError('Configuration must be numpy ndarray of size 6')
        if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
            raise ValueError('Illegal vacuum axis. Must be norm one')
        return configuration[0:3], configuration[3:6]


class CylinderPoint3D(PointGrasp):
    """圆柱抓手点"""

    def __init__(self, configuration, frame='object', grasp_id=None):
        # get parameters from configuration array
        grasp_center, grasp_axis, grasp_depth, max_grasp_width, min_grasp_width = \
            CylinderPoint3D.params_from_configuration(configuration)

        self.center_ = grasp_center
        self.axis_ = grasp_axis / np.linalg.norm(grasp_axis)
        self.max_grasp_width_ = max_grasp_width
        self.min_grasp_width_ = min_grasp_width
        self.grasp_depth_ = grasp_depth
        self.frame_ = frame
        self.grasp_id_ = grasp_id
    
    @property
    def center(self):
        """ :obj:`numpy.ndarray` : 3-vector specifying the center of the jaws """
        return self.center_

    @center.setter
    def center(self, x):
        self.center_ = x

    @property
    def axis(self):
        """ :obj:`numpy.ndarray`---表示圆柱轴的3d向量 """
        return self.axis_

    @property
    def open_width(self):
        """ float : maximum opening width of the jaws """
        return self.max_grasp_width_

    @property
    def close_width(self):
        """ float : minimum opening width """
        return self.min_grasp_width_

    @property
    def open_width(self):
        """ float : maximum opening width """
        return self.max_grasp_width_

    @property
    def approch_depth(self):
        """ float : approach angle of the grasp """
        return self.grasp_depth_
    
    @property
    def configuration(self):
        """ :obj:`numpy.ndarray` : vector specifying the parameters of the grasp as follows
        (grasp_center, grasp_axis, grasp_angle, grasp_width, jaw_width) """
        return CylinderPoint3D.configuration_from_params(self.center_, self.axis_, self.max_grasp_width_, self.min_grasp_width_, 
                                                              self.grasp_depth_)

    @property
    def frame(self):
        """ :obj:`str` : name of grasp reference frame """
        return self.frame_

    @property
    def id(self):
        """ int : id of grasp """
        return self.grasp_id_

    @frame.setter
    def frame(self, f):
        self.frame_ = f

    @approch_depth.setter
    def approach_depth(self, depth):
        """ Set the grasp approach angle """
        self.grasp_depth_ = depth

    @staticmethod
    def axis_from_endpoints(g1, g2):
        """ Normalized axis of grasp from endpoints as np 3-arrays """
        grasp_axis = g2 - g1
        if np.linalg.norm(grasp_axis) == 0:
            return grasp_axis
        return grasp_axis / np.linalg.norm(grasp_axis)

    @property
    def endpoints(self):
        """
        Returns
        -------
        :obj:`numpy.ndarray 36X3`
            location of endpoints in 3D space at max opening width "with approch depth" """
        def rotate_mat(axis, radian):
            rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
            return rot_matrix
        
        vertical_array = np.cross(self.axis_, np.array([1,0,0]))
        endpoints_set = np.zeros(36,3)
        for i in range(36):
            R = rotate_mat(vertical_array, i*10*np.pi/180)
            endpoints_set[i,:] = self.center_+ self.axis_ * self.grasp_depth_ + vertical_array.dot(R) * self.max_grasp_width_

        return endpoints_set
    
    @staticmethod
    def distance(grasp1, grasp2, alpha=0.05):
        """ Evaluates the distance between two grasps.

        Returns
        -------
        float
            distance between grasps g1 and g2
        """
        end1 = grasp1.center + grasp1.axis * grasp1.approch_depth
        end2 = grasp2.center + grasp2.axis * grasp2.approch_depth
        end_dist = np.linalg.norm(end1- end2)
        axis_dist = (2.0 / np.pi) * np.arccos(np.abs(grasp1.axis.dot(grasp2.axis)))
        return end_dist + alpha * axis_dist

    @staticmethod
    def params_from_configuration(configuration):
        """ Converts configuration vector into grasp parameters.
        # grasp_center, grasp_axis, grasp_depth, max_grasp_width, min_grasp_width
        Returns
        -------
        grasp_center : :obj:`numpy.ndarray`
            center of grasp in 3D space
        grasp_axis : :obj:`numpy.ndarray`
            normalized axis of grasp in 3D space
        grasp_depth : float
            maximum opening width of jaws
        max_grasp_width : float
            approach angle
        min_grasp_width : float
            minimum closing width of jaws
        """
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 8 and configuration.shape[0] != 9):
            raise ValueError('Configuration must be numpy ndarray of size 8 or 9')
        if configuration.shape[0] == 8:
            min_grasp_width = 0
        else:
            min_grasp_width = configuration[8]
        if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        return configuration[0:3], configuration[3:6], configuration[6], configuration[7], min_grasp_width

    @staticmethod
    def configuration_from_params(grasp_center, grasp_axis, grasp_depth, max_grasp_width, min_grasp_width=0):
        """ Converts grasp parameters to a configuration vector. """
        if np.abs(np.linalg.norm(grasp_axis) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        configuration = np.zeros(9)
        configuration[0:3] = grasp_center
        configuration[3:6] = grasp_axis
        configuration[6] = grasp_depth
        configuration[7] = max_grasp_width
        configuration[8] = min_grasp_width
        return configuration

    @property
    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the 0-degree approach direction, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
        if np.linalg.norm(grasp_axis_x) == 0:
            grasp_axis_x = np.array([1, 0, 0])
        grasp_axis_x = grasp_axis_x / norm(grasp_axis_x)
        grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        return R
    
    @property
    def rotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the grasp approach angle, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        R = CylinderPoint3D._get_rotation_matrix_y(self.approach_angle)
        R = self.unrotated_full_axis.dot(R)
        return R

    @property
    def T_grasp_obj(self):
        """ Rigid transformation from grasp frame to object frame.
        Rotation matrix is X-axis along approach direction, Y axis pointing between the jaws, and Z-axis orthogonal.
        Translation vector is the grasp center.

        Returns
        -------
        :obj:`RigidTransform`
            transformation from grasp to object coordinates
        """
        T_grasp_obj = RigidTransform(self.rotated_full_axis, self.center, from_frame='grasp', to_frame='obj')
        return T_grasp_obj

    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R

    def gripper_pose(self, gripper=None):
        """ Returns the RigidTransformation from the gripper frame to the object frame when the gripper is executing the
        given grasp.
        Differs from the grasp reference frame because different robots use different conventions for the gripper
        reference frame.
        
        Parameters
        ----------
        gripper : :obj:`RobotGripper`
            gripper to get the pose for

        Returns
        -------
        :obj:`RigidTransform`
            transformation from gripper frame to object frame
        """
        if gripper is None:
            T_gripper_grasp = RigidTransform(from_frame='gripper', to_frame='grasp')
        else:
            T_gripper_grasp = gripper.T_grasp_gripper

        T_gripper_obj = self.T_grasp_obj * T_gripper_grasp
        return T_gripper_obj



    def close_fingers( center, axis, grasp_depth, radius, obj, vis=False, check_approach=True, approach_dist=1.0):
        """ Steps along grasp axis to find the locations of contact with an object
        `world frame`  return `grid frame`

        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to close fingers on
        vis : bool
            whether or not to plot the line of action and contact points
        check_approach : bool
            whether or not to check if the contact points can be reached
        approach_dist : float
            how far back to check the approach distance, only if checking the approach is set
        
        Returns
        -------
        success : bool
            whether or not contacts were found
        c1 : :obj:`Contact3D`
            the contact point for jaw 1
        c2 : :obj:`Contact3D`
            the contact point for jaw 2
        """
        if vis:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()
        # compute num samples to use based on sdf resolution
        grasp_radius_grid = obj.sdf.transform_pt_obj_to_grid(float(radius))
        grasp_depth_grid = obj.sdf.transform_pt_obj_to_grid(float(grasp_depth))
        r_num_samples = int(Grasp.samples_per_grid * float(grasp_radius_grid))  # at least 1 sample per grid
        d_num_samples = int(Grasp.samples_per_grid * float(grasp_depth_grid)) *2
        r_num_samples = max(r_num_samples,3)
        d_num_samples = max(d_num_samples,10)

        points_of_action, collision = CylinderPoint3D.create_cylinder_of_action(center, axis, grasp_depth, radius, obj,
                                                            r_num_samples, d_num_samples)
        

        contacts = np.zeros([d_num_samples* 36]).tolist()
        # check for contact along approach
        if check_approach:#检测张开时是否有碰撞
            if collision == True:
                print("collision")
                return False, None
            # for i in np.arange(0, 36):
            #     line_of_approach = points_of_action[:,int(r_num_samples)-1,i,:]
            #     c_apprach_found, _ = CylinderPoint3D.find_contact(list(line_of_approach), obj, vis=vis, strict=True)
            #     if c_apprach_found:
            #         print("collision!!")
            #         return False, None
            # for i in np.arange(0, 36):
            #     line_of_approach = points_of_action[0,:,i,:] - np.repeat([axis],int(r_num_samples), axis=0 )*(0.2-grasp_depth)
            #     c_apprach_found_r, _ = CylinderPoint3D.find_contact(list(line_of_approach), obj, vis=vis, strict=True)
            #     if c_apprach_found_r:
            #         print("collision")
            #         return False, None

        contacts_found_ = False
        for i in np.arange(0, 36):
            for j in np.arange(0, int(d_num_samples)):
                line_of_action = points_of_action[j,:,i,:]
                c_found, c = CylinderPoint3D.find_contact(list(line_of_action), obj, vis=vis)
                contacts_found_ = contacts_found_ or c_found
                contacts[j*36+i] = c
        

        return contacts_found_, contacts

    @staticmethod
    def create_cylinder_of_action(center, axis, depth, radius, obj, r_num_samples, d_num_samples, min_width=0, convert_grid=True):
        """
        `world frame` return `grid frame`

        Creates a straight line of action, or list of grid points, from a given point and direction in world or grid coords

        Parameters
        ----------
        center : 3x1 :obj:`numpy.ndarray`
            start point to create the line of action
        axis : normalized 3x1 :obj:`numpy.ndarray`
            normalized numpy 3 array of grasp direction
        width : float
            the grasp width
        num_samples : int
            number of discrete points along the line of action
        convert_grid : bool
            whether or not the points are specified in world coords

        Returns
        -------
        points_of_action : :obj:`list` of d_num_samples X 36 X r_num_samples X3
        """
        def rotate_mat(axis, radius):
            rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radius))
            return rot_matrix
        def cylinder_points(center, axis, depth, radius):          
            vertical_array = np.cross(axis, np.array([1,0,0]))
            vertical_array = vertical_array / np.linalg.norm(vertical_array)
            endpoints_set = np.zeros([36,3])
            for i in np.arange(0, 36):
                R = rotate_mat(vertical_array, i*10*np.pi/180)
                endpoints_set[i,:] = center + axis * depth + vertical_array.dot(R) * radius
            return endpoints_set
        
        r_num_samples = max(r_num_samples, 3)  # always at least 3 samples
        d_num_samples = max(d_num_samples, 10)  # always at least 10 samples
        points_of_action = np.zeros([d_num_samples, r_num_samples, 36, 3])
        radius_sets = np.linspace(float(radius),0, num=r_num_samples)
        depth_sets = np.linspace(0, float(depth), num=d_num_samples)
        print(points_of_action.shape)
        # points_of_action = np.zeros([d_num_samples, r_num_samples, 36, 3]).tolist()
        vertical_array = np.cross(axis, np.array([1,0,0]))
        vertical_array = vertical_array / np.linalg.norm(vertical_array)

        # # check bottom
        # for i in np.arange(0, 36):
        #     R = rotate_mat(vertical_array, i*10*np.pi/180)
        #     for r_index, r in enumerate(radius_sets):
        #         bottom_of_action = np.zeros([r_num_samples, 36, 3])
        #         # points_of_action[0,:,i,:] - np.repeat([axis],int(r_num_samples), axis=0 )*(0.2-grasp_depth)
        #         bottom_of_action[r_num_samples-1-r_index,i,:] = center - axis * (0.2-depth) + vertical_array.dot(R) * r
        #         if convert_grid:
        #             as_array = bottom_of_action[r_num_samples-1-r_index,i,:].T # 3x36
        #             # print(as_array.shape)
        #             transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
        #             bottom_of_action[r_num_samples-1-r_index,i,:] = transformed.T
        #     line_of_approach = bottom_of_action[:,i,:]
        #     c_apprach_found, _ = CylinderPoint3D.find_contact(list(line_of_approach), obj, strict=True)
        #     if c_apprach_found == True:
        #         print("collision for bottom!!")
        #         return None, True

        for r_index, r in enumerate(radius_sets):
            for i in np.arange(0, 36):         
                R = rotate_mat(vertical_array, i*10*np.pi/180)
                for d_index, d in enumerate(depth_sets):
                    points_of_action[d_index,r_num_samples-1-r_index,i,:] = center + axis * d + vertical_array.dot(R) * r

                    if convert_grid:
                        as_array = points_of_action[d_index,r_num_samples-1-r_index,i,:].T # 3x36
                        # print(as_array.shape)
                        transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
                        points_of_action[d_index,r_num_samples-1-r_index,i,:] = transformed.T

                # if  r_index == 0:
                #     line_of_approach = points_of_action[:,r_num_samples-1,i,:]
                #     c_apprach_found, _ = CylinderPoint3D.find_contact(list(line_of_approach), obj, strict=True)
                #     if c_apprach_found == True:
                #         print("collision for approaching!!")
                #         return None, True
        collision = False
        return points_of_action, collision
    
    @staticmethod
    def find_contact(line_of_action, obj, vis=False, strict=False):
        """
        `grid frame` return `world frame`

        Find the point at which a point traveling along a given line of action hits a surface.

        Parameters
        ----------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            the points visited as the fingers close (grid coords)
        obj : :obj:`GraspableObject3D`
            to check contacts on
        vis : bool
            whether or not to display the contact check (for debugging)

        Returns
        -------
        contact_found : bool
            whether or not the point contacts the object surface
        contact : :obj:`Contact3D`
            found along line of action (None if contact not found)
        """
        contact_found = False
        pt_zc = None
        pt_zc_world = None
        contact = None
        num_pts = len(line_of_action)
        sdf_here = 0
        sdf_before = 0
        pt_grid = None
        pt_before = None

        # step along line of action, get points on surface when possible
        i = 0
        while i < num_pts and not contact_found:
            # update loop vars
            pt_before_before = pt_before
            pt_before = pt_grid
            sdf_before_before = sdf_before
            sdf_before = sdf_here
            pt_grid = line_of_action[i]

            # visualize
            if vis:
                ax = plt.gca(projection='3d')
                ax.scatter(pt_grid[0], pt_grid[1], pt_grid[2], c='r')

            # check surface point
            on_surface, sdf_here = obj.sdf.on_surface(pt_grid)
            if on_surface:
                contact_found = True
                if strict:
                    return contact_found, None

                # quadratic approximation to find actual zero crossing
                if i == 0:
                    pt_after = line_of_action[i + 1]
                    sdf_after = obj.sdf[pt_after]
                    pt_after_after = line_of_action[i + 2]
                    sdf_after_after = obj.sdf[pt_after_after]

                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_grid, sdf_here, pt_after, sdf_after, pt_after_after,
                                                               sdf_after_after)
                    # contact not yet found if next sdf value is smaller
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False

                elif i == len(line_of_action) - 1:
                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_before_before, sdf_before_before, pt_before,
                                                               sdf_before, pt_grid, sdf_here)
                    if pt_zc is None:
                        contact_found = False
                else:
                    pt_after = line_of_action[i + 1]
                    sdf_after = obj.sdf[pt_after]
                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_before, sdf_before, pt_grid, sdf_here, pt_after,
                                                               sdf_after)
                    # contact not yet found if next sdf value is smaller
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False
            i = i + 1


        if contact_found:
            pt_zc_world = obj.sdf.transform_pt_grid_to_obj(pt_zc)
            in_direction_grid = line_of_action[-1] - line_of_action[0]
            in_direction_grid = in_direction_grid / np.linalg.norm(in_direction_grid)
            in_direction = obj.sdf.transform_pt_grid_to_obj(in_direction_grid, direction=True)
            contact = Contact3D(obj, pt_zc_world, in_direction=in_direction)
            if contact.normal is None:
                contact_found = False
        return contact_found, contact

    @staticmethod #grasp_center, grasp_axis, grasp_depth, max_grasp_width, min_grasp_width
    def grasp_from_contact_and_axispoint_on_grid(obj, grasp_center_world, grasp_another_world, grasp_depth_world, max_grasp_width_world, 
                                            min_grasp_width_world=0, vis=False, backup=0.5):

        # transform to grid basis
        grasp_axis_world = CylinderPoint3D.axis_from_endpoints( grasp_center_world,grasp_another_world)
        grasp_axis_world = grasp_axis_world / np.linalg.norm(grasp_axis_world)
        grasp_axis_grid = obj.sdf.transform_pt_obj_to_grid(grasp_axis_world, direction=True)
        max_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(max_grasp_width_world)
        min_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(min_grasp_width_world)
        grasp_depth_grid = obj.sdf.transform_pt_obj_to_grid(grasp_depth_world)
        grasp_center_grid = obj.sdf.transform_pt_obj_to_grid(
            grasp_center_world) ###- backup * grasp_axis_grid  # subtract to find true point
        r_num_samples = int(2 * max_grasp_width_grid)  # at least 2 samples per grid
        d_num_samples = int(2 * grasp_depth_grid)


        is_contacts_found, contacts_found = CylinderPoint3D.close_fingers( grasp_center_world, grasp_axis_world, grasp_depth_world, 
                                                                max_grasp_width_world, obj, vis=False, check_approach=True, approach_dist=1.0)

        if not is_contacts_found:
            print('No contacts found for grasp')
            return None, None

        contacts_final = list()
        contacts_found = [i for n, i in enumerate(contacts_found) if i not in contacts_found[:n]]
        if 0 in contacts_found:
            contacts_found.remove(0)

        surface_points, _ = obj.sdf.surface_points(grid_basis=False)
        for i, x_surf in enumerate(surface_points):
            for j, contact in enumerate(contacts_found):
                if contact != None:
                    # print(x_surf)
                    # print(contact.point_)
                    if np.linalg.norm(x_surf - np.array(contact.point_)) < 5 * obj.sdf.resolution:
                        contacts_final.append(contact) 
                        break

        configuration = CylinderPoint3D.configuration_from_params(grasp_center_world, grasp_axis_world, grasp_depth_world,
                                                                       max_grasp_width_world)

        print("find contacts:",len(contacts_final))
        # print(configuration)
        grasp = CylinderPoint3D(configuration)
        # print("grasp: ",grasp)

        if (len(contacts_final) > 0) and (grasp is not None):
            return grasp, contacts_final # relative to object
        else:
            return None, None


    @staticmethod #grasp_center, grasp_axis, grasp_depth, max_grasp_width, min_grasp_width
    def grasp_from_contact_and_axis_on_grid(obj, grasp_center_world, grasp_axis_world, grasp_depth_world, max_grasp_width_world, 
                                            min_grasp_width_world=0, vis=False, backup=0.5):
    
        grasp_axis_world = grasp_axis_world / np.linalg.norm(grasp_axis_world)
        grasp_axis_grid = obj.sdf.transform_pt_obj_to_grid(grasp_axis_world, direction=True)
        max_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(max_grasp_width_world)
        min_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(min_grasp_width_world)
        grasp_depth_grid = obj.sdf.transform_pt_obj_to_grid(grasp_depth_world)
        grasp_center_grid = obj.sdf.transform_pt_obj_to_grid(
            grasp_center_world) ###- backup * grasp_axis_grid  # subtract to find true point
        r_num_samples = int(2 * max_grasp_width_grid)  # at least 2 samples per grid
        d_num_samples = int(2 * grasp_depth_grid)

        is_contacts_found, contacts_found = CylinderPoint3D.close_fingers( grasp_center_world, grasp_axis_world, grasp_depth_world, 
                                                                max_grasp_width_world, obj, vis=False, check_approach=True, approach_dist=1.0)

        if not is_contacts_found:
            print('No contacts found for grasp')
            return None, None

        contacts_final = list()
        contacts_found = [i for n, i in enumerate(contacts_found) if i not in contacts_found[:n]]
        if 0 in contacts_found:
            contacts_found.remove(0)

        surface_points, _ = obj.sdf.surface_points(grid_basis=False)
        for i, x_surf in enumerate(surface_points):
            for j, contact in enumerate(contacts_found):
                if contact != None:
                    # print(x_surf)
                    # print(contact.point_)
                    if np.linalg.norm(x_surf - np.array(contact.point_)) < 5 * obj.sdf.resolution:
                        contacts_final.append(contact) 
                        break

        configuration = CylinderPoint3D.configuration_from_params(grasp_center_world, grasp_axis_world, grasp_depth_world,
                                                                       max_grasp_width_world)

        print("find contacts:",len(contacts_final))
        # print(configuration)
        grasp = CylinderPoint3D(configuration)
        # print("grasp: ",grasp)

        if (len(contacts_final) > 0) and (grasp is not None):
            return grasp, contacts_final # relative to object
        else:
            return None, None


    @staticmethod
    def generate_grasps(obj, max_depth, radius):
        def rotate_mat(axis, radian):
            return linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        def cylinder_points(center, axis, depth, radius):          
            vertical_array = np.cross(axis, np.array([1,0,0]))
            vertical_array = vertical_array / np.linalg.norm(vertical_array)
            endpoints_set = np.zeros([360,3])
            for i in np.arange(0, 360):
                R = rotate_mat(axis, i*1*np.pi/180)
                endpoints_set[i,:] = center + axis * depth + vertical_array.dot(R) * radius
            return endpoints_set

        # 抓取轴
        grasp_axis = np.random.random(3) * 2 -1
        grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)

        # 找抓取点
        surface_points, _ = obj.sdf.surface_points(grid_basis=False)
        min_projected_distance = float('inf')
        grasp_center = None
        for x_surf in surface_points:
            # print((x_surf))
            # as_array = x_surf.T # 3x36
            # transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
            # x_surf = transformed.T
            
            projected_distance = x_surf.dot(grasp_axis) 
            if  projected_distance < min_projected_distance:
                min_projected_distance = projected_distance
                grasp_center = x_surf

        # # 确定深度
        # grasp_depth = 0
        # while grasp_depth < max_depth:
        #     endpoints_set = cylinder_points(grasp_center, grasp_axis, grasp_depth, radius)
            

        #     count = 0
        #     for end_point in endpoints_set:
        #         # print((end_point))
        #         # on_surf,_ = obj.sdf._signed_distance(end_point)
        #         # print(signed_distance)
        #         # print(np.linalg.norm(end_point-(grasp_center+grasp_axis*grasp_depth)))
        #         end_point = obj.sdf.transform_pt_obj_to_grid(end_point.T).T
        #         # a = obj.sdf.transform_pt_obj_to_grid((grasp_center+grasp_axis*grasp_depth).T).T
        #         # print(np.linalg.norm(end_point-a))
        #         # print(np.linalg.norm(grasp_axis*grasp_depth))
        #         if obj.sdf._signed_distance(end_point) < -0.001:
        #             # print("on_surf:",on_surf)
        #             count += 1
        #             if count > 10:
        #                 print("grasp_depth:",grasp_depth)
        #                 return grasp_center, grasp_axis, grasp_depth, radius  
        #     grasp_depth += max_depth / 100
        # print("grasp_depth:",grasp_depth)
        # return grasp_center, grasp_axis, grasp_depth, radius
                # 确定深度
        grasp_depth = 0
        while grasp_depth < max_depth:
            wrapped_surf = []
            for x_surf in surface_points:
                if abs((x_surf-grasp_center).dot(grasp_axis) - grasp_depth)<0.0001:
                    if np.linalg.norm(x_surf-(grasp_center+grasp_axis*grasp_depth)) > radius :
                        print("grasp_depth:",grasp_depth)
                        return grasp_center, grasp_axis, grasp_depth, radius , wrapped_surf
                elif (x_surf-grasp_center).dot(grasp_axis) < grasp_depth:
                    wrapped_surf.append(x_surf)
            grasp_depth += max_depth / 100
        print("grasp_depth:",grasp_depth)
        return grasp_center, grasp_axis, grasp_depth, radius, wrapped_surf






