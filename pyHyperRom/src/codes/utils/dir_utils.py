from .basic import *

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)
