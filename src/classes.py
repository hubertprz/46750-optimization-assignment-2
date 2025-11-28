import pandas as pd
class Substation():
    def __init__(self, id, node, capacity, lines: list):
        self.id = id
        self.node = node
        self.capacity = capacity
        self.lines = lines

class Heatpump():
    def __init__(self, id, node, load):
        self.id = id
        self.node = node
        self.load = load

class EV():
    def __init__(self, id, node, load):
        self.id = id
        self.node = node
        self.load = load

class DistributionNetwork():
    """
    Class representing a Distribution Network.

    Attributes:
        NODES (list): List of existing nodes.
        LOADS (list): List of existing system nodes
        SUBSTATIONS (list): List of existing substations.
        load_capacity (dict): Loads capacity (P^D_i)
        substation_capacity (dict): Substation capacity
        mapping_lines (dict): Matrix of connections between nodes ij (=1 if connected, = 0 if not connected)
        mapping_loads (dict): Matrix of location of loads at nodes (...)
        mapping_substations (dict): Matrix of location of substations at nodes (=1 if generator is at node i, = 0 if not at node i)
        branch_capacity (pd.DataFrame): Matrix of susceptance of distribution lines between nodes ij
        bus_susceptance (pd.DataFrame): Matrix B of bus susceptance
    Methods:
    """
    def __init__(self, NODES, LOADS, SUBSTATIONS, load_capacity, substation_capacity, mapping_lines, mapping_loads, mapping_substations):
        self.NODES = NODES
        self.LOADS = LOADS
        self.SUBSTATIONS = SUBSTATIONS
        self.load_capacity = load_capacity
        self.substation_capacity = substation_capacity
        self.mapping_lines = mapping_lines
        self.mapping_loads = mapping_loads
        self.mapping_substations = mapping_substations
        #branch_capacity: pd.DataFrame
        #bus_susceptance: pd.DataFrame
        #slack_bus: str

    def add_candidate_substations(self, substations):
        """Adds candidate substations for expansion."""
        self.substation_candidates = []
        if not isinstance(substations, list):
            substations = [substations]
        else:
            for S in substations:
                self.substation_candidates.append(S.id)
                self.mapping_substations.loc[S.id, S.node] = 1

    def add_appliances(self, appliances, nodes):
        if not isinstance(appliances, list):
            appliances = []
        else:
            pass # TO DO...