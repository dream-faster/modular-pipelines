from copy import deepcopy

from utils.flatten import flatten


class Example:
    def __init__(self, id, children=[]):
        self.id = id  # np.randint(50, 100)
        self.children = children

    def get_children(self):
        return (
            [self] + self.children + [child.get_children() for child in self.children]
        )


child1 = Example(111)
child2 = Example(112)
child3 = Example(113)
parent = Example(11, [child1, child2, child3])
grandparent = Example(1, [parent])

all_children = flatten(grandparent.get_children())


def create_hierarchical_dict(pipeline) -> dict:
    entire_pipeline = pipeline.get_children()
    parent = ""

    def get_child(block, parent):
        new_dict = dict()
        if isinstance(block, list):

            new_dict[parent] = {
                "name": parent,
                "children": [get_child(child, parent) for child in block],
            }
        else:
            new_dict[block.id] = block.id
            parent = block.id

        return new_dict

    hierarchy = get_child(entire_pipeline, "hierarchy")
    print(hierarchy)
    return hierarchy


create_hierarchical_dict(grandparent)
