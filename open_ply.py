'''
Description: 
Author: Bin Peng
Date: 2023-05-08 15:36:21
LastEditTime: 2023-05-08 22:00:33
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

def parse_ply(fname):
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
		'vertex_indices': (lambda f, l, m=m: f.set_vertices(l))                            #references to vertices
	}
	element_type_dict = {
		'vertex' : (lambda: Mesh.Vertex(), vertex_names, lambda v, m=m: m.add_vertex(v)),
		'face' : (lambda: Mesh.Face(), face_names, lambda f, m=m: m.add_face(f))
	}
	type_handles = {
		'float' : get_float,
		'double' : get_double,
		'char' : get_char,
		'uchar' : get_uchar,
		'short' : get_short,
		'ushort' : get_ushort,
		'int' : get_int,
		'uint' : get_uint
	}
	i = 0
	j = 0
	for line in open(fname, 'r'):
		line = line.rstrip()
		if state == 'init':
			if line != 'ply':
				raise RuntimeError('PLY: file is not a ply file')
			state = 'format'
		elif state == 'format':
			if not format_re.match(line):
				raise RuntimeError('PLY: unsupported ply format')
			state = 'header'
		elif state == 'header':
			if comment_re.match(line):
				#comment, do nothing
				continue
			match = element_re.match(line)
			if match:
				element_types.append((match.group('name'), int(match.group('num')), []))
				continue
			match = property_list_re.match(line)
			if match:
				element_types[-1][2].append((match.group('name'), 'list', match.group('itype'), match.group('etype')))
				continue
			match = property_re.match(line)
			if match:
				element_types[-1][2].append((match.group('name'), match.group('type')))
				continue
			if line == 'end_header':
				state = 'body'
				continue
			raise RuntimeError('PLY: unknown header field')
		elif state == 'body':
			if j >= element_types[i][1]:
				j = 0
				i = i + 1
			if i >= len(element_types):
				raise RuntimeExeception('PLY: too much data in file')
			line = line.split()
			actions = element_type_dict[element_types[i][0]]
			obj = actions[0]()
			for property in element_types[i][2]:
				x = None
				if property[1] == 'list':
					numelems = type_handles[property[2]](line[0])
					line = line[1:]
					x = []
					for count in range(numelems):
						x.append(type_handles[property[3]](line[0]))
						line = line[1:]
				else:
					x = type_handles[property[1]](line[0])
					line = line[1:]
				actions[1][property[0]](obj, x)
			actions[2](obj)
			j = j + 1
	return m