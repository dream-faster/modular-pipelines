from utils.dict import obj_to_dict, flatten
import pandas as pd
import numpy as np


def get_random_obj():
    class RandomChild:
        def __init__(self, i):
            self.id = "child " + str(i)
            self.integer = 5
            self.string = "hello this is a string"

    class ChildwithChildren:
        def __init__(self):
            self.id = "child_with_children"
            self.random_children = [RandomChild(i) for i in range(5)]

    class RandomParentClass:
        def __init__(self):
            self.id = "random_parent_class"
            self.random_children = [RandomChild(i) for i in range(5)]
            self.integer = 5
            self.string = "hello this is a string"
            self.child_with_children = ChildwithChildren()
            self.df = pd.DataFrame([[1, 2, 3], ["a", "b", "c"]])
            self.a_numpy = np.array([1, 2, 3])

    return RandomParentClass()


def test_obj_to_dict():
    random_obj = get_random_obj()
    for k, v in flatten(obj_to_dict(random_obj, type_exclude="str")).items():
        assert isinstance(k, str), f"{k} is not a string"
        assert isinstance(v, (str, int, float)), f"{v} is not a string, int or float"
