from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees import logging as log_tree


class Orchestrator:
    def __init__(self):
        root = Sequence(name="sequence", memory=True)