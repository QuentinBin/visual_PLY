'''
Description: 
Author: Bin Peng
Date: 2023-05-08 15:36:21
LastEditTime: 2023-05-08 16:35:16
'''
import re

import Mesh


def get_float(str):
	return float(str)

def get_double(str):
	return float(str)

def get_char(str):
	return int(str)

def get_uchar(str):
	return int(str)

def get_short(str):
	return int(str)

def get_ushort(str):
	return int(str)

def get_int(str):
	return int(str)

def get_uint(str):
	return int(str)

def parse_ply(f_name):
	m = Mesh.Mesh()
	state = 'init'
	format_re = re.compile('format\\s+ascii\\s+1.0')
	comment_re = re.compile('comment\\s.*')
	element_re = re.compile('element\\s+(?P<name>\\w+)\\s+(?P<num>\\d+)')
	property_re = re.compile('property\\s+(?P<type>\\w+)\\s+(?P<name>\\w+)')
	property_list_re = re.compile('property\\s+list\\s+(?P<itype>\\w+)\\s+(?P<etype>\\w+)\\s+(?P<name>\\w+)')
	element_types = []
	vertex_names = {
		'x': lambda v, x: v.set_X(x),
		'y': lambda v, y: v.set_Y(y),
		'z': lambda v, z: v.set_Z(z),
		'nx': lambda v, nx: v.set_NX(nx),
		'ny': lambda v, ny: v.set_NY(ny),
		'nz': lambda v, nz: v.set_NZ(nz),
		's': lambda v, s: v.set_S(s),
		't': lambda v, t: v.set_T(t)
	}
	face_names = {
		#'vertex_indices': (lambda f, l, m=m: f.set([m.getVertex(v) for v in l])) #real values of vertices
		'vertex_indices': (lambda face, l, m=m: face.set_vertices(l))   # indices pf vertices                         #references to vertices
	}
	