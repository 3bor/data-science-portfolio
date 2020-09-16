import pkg_resources
import types
import sys

def print_version_info(module_globals):
	"""
	print_version_info(globals()) -- prints python and package version info
	"""
	# Print python version
	print("python=={}.{}".format(sys.version_info.major,sys.version_info.minor))

	imports = list(set(get_imports(module_globals)))

	requirements = []
	for m in pkg_resources.working_set:
		if m.project_name in imports and m.project_name!="pip":
			requirements.append((m.project_name, m.version))
			
	# Print package versions		
	for r in requirements:
		print("{}=={}".format(*r))
    
    
def get_imports(module_globals):
	for name, val in module_globals.items():
		if isinstance(val, types.ModuleType):
			# Split ensures you get root package, 
			# not just imported function
			name = val.__name__.split(".")[0]
			
		elif isinstance(val, type):
			name = val.__module__.split(".")[0]
			
		# Some packages are weird and have different
		# imported names vs. system names
		if name == "PIL":
			name = "Pillow"
		elif name == "sklearn":
			name = "scikit-learn"
		yield name