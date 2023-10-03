import re as reg

class Attribute_dict:
	'''
	recursively convert dictionary to an object where keys are attributes
	'''
	def __init__(self,dict_in):
		for k in dict_in:
			v = dict_in[k]
			if type(v) is dict:
				self.__dict__[k] = Attribute_dict(v)
			else:
				self.__dict__[k] = v

	def update(self,dict_in):
		for k in dict_in:
			v = dict_in[k]
			self.__dict__[k] = Attribute_dict(v) if type(v) is dict else v

	def dict(self):
		out_dict = {}
		for k in self.__dict__:
			v = self.__dict__[k]
			out_dict[k] = v.dict() if type(v) is Attribute_dict else v
		return out_dict

	def __str__(self,indent='',node_indent=''):
		local_indent = '  |  '
		local_node_indent = '  |--'
		str_out = ''
		for k in self.__dict__:
			v = self.__dict__[k]
			if type(v) is Attribute_dict:
				str_out += node_indent + k+': \n'
				contents_str = v.__str__(indent=indent+local_indent,node_indent=indent+local_node_indent)
				contents_str = contents_str
				if not contents_str == '': # don't print empty line for empty object
					str_out += contents_str+'\n'
			else:
				str_out += node_indent
				str_out += '{}: {}\n'.format(k,v)

		return str_out[:-1]
