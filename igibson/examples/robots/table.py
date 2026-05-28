from igibson.objects.articulated_object import URDFObject
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from pprint import pprint

import os

import inspect

# All public attributes on the class (incl. methods, properties)
print([k for k in dir(URDFObject) if not k.startswith("_")])

# Just methods defined on the class
print([name for name, fn in inspect.getmembers(URDFObject, inspect.isfunction)
       if not name.startswith("_")])
