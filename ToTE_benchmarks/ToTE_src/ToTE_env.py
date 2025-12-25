import os
import json
import sys

from .linear_simulator import LinearSimulator

sys.path.append(os.path.join('..', '..'))

from networkx.readwrite import json_graph
from .utils import paths_from_file

from src.config import DATA_DIR
import numpy as np


class ToTEEnv(object):

    def __init__(self, props):
        self.topo_name = props.topo_name
        self.props = props
        self.init_topo()
        self.simulator = LinearSimulator(props)


    def init_topo(self):
        self.G = self.read_graph_json(self.topo_name)
        self.calculate_graph_stats()
        self.load_paths()
        self.initialize_capacity_matrix()

    def calculate_graph_stats(self):
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()

    def load_paths(self):
        paths_file_path = f"{DATA_DIR}/{self.topo_name}/{self.props.paths_file}"
        if self.props.paths_file=="no_path":
            self.pij = None
        else:
            self.pij = paths_from_file(paths_file=paths_file_path, num_nodes=self.num_nodes)

    def initialize_capacity_matrix(self):
        self.G.capacity_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in self.G.adj:
            for j, edge_data in self.G.adj[i].items():
                if 'capacity' in edge_data:
                    self.G.capacity_matrix[i][j] = edge_data['capacity']
                else:
                    pass

    def read_graph_json(self, topo):
        with open(os.path.join(DATA_DIR, topo, topo + '.json'), 'r') as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)
