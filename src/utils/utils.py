'''
Author: Jiaxin Zheng
Date: 2024-04-03 20:27:39
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 21:51:24
Description: 
'''
import importlib
def is_number(s):
    """Check whether it is a number"""
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def create_object(object_path):
    module_path, object_name = object_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    object = getattr(module, object_name)
    
    return object
