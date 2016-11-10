
'''
A few utils useful for online nodes.
'''

#TODO: Have to merge with the main utils at some point.


def update_dict_lists(src, dst):
    [dst[key].append(src[key]) for key in dst.keys()]
    return dst


