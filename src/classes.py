import pandas as pd
class Substation:
    def __init__(self, id, node, capacity, neighbors, fix_cost = 0, edge_cost = 0):
        self.id = id
        self.node = node
        self.capacity = capacity
        self.neighbors = neighbors  # Nodes with potential connection
        self.fix_cost = fix_cost    # Fixed cost for activation
        self.edge_cost = edge_cost  # Cost for activating arc connection

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
        SUBSTATIONS (list): List of existing substations (Substation objects).
        load_capacity (dict): Loads capacity (P^D_i)
        substation_capacity (dict): Substation capacity

    Methods:
        add_candidate_substations:
        update_arcs:
        update_substations:
    """
    def __init__(self, NODES, LOADS, SUBSTATIONS, load_capacity, nodes_connected, loads_locations):
        self.NODES = NODES
        self.LOADS = LOADS
        self.SUBSTATIONS = SUBSTATIONS
        self.load_capacity = load_capacity
        self.nodes_connected = nodes_connected
        self.loads_locations = loads_locations

        # Arcs (distribution lines)
        self.update_arcs()
        self.edge_cost = {e: 0 for e in self.E} # Cost of initial lines set to 0

        self.mapping_substations = pd.DataFrame()
        for S in self.SUBSTATIONS:
            self.mapping_substations.loc[S.id, self.NODES] = 0
            self.mapping_substations.loc[S.id, S.node] = 1

        #branch_capacity: pd.DataFrame
        #bus_susceptance: pd.DataFrame
        #slack_bus: str

    def add_candidate_substations(self, substations):
        """Adds candidate substations for expansion."""
        if not isinstance(substations, list):
            substations = [substations]
        
        for S in substations:
            self.SUBSTATIONS.append(S)
            self.NODES.append(S.node)
            self.nodes_connected[S.node] = S.neighbors
            
            self.update_arcs()
            for neighbor in S.neighbors:
                self.nodes_connected[neighbor].append(S.node) # Also adding connection in nodes that existed previously

                a = int(S.node[1:])             # Index (integer) of the node
                b = int(neighbor[1:])           # Index (integer) of the neighboring node
                e = ((min(a,b), max(a,b)))      # Arc (from lower to higher index)
                self.edge_cost[e] = S.edge_cost # Setting edge cost to that arc

    def add_appliances(self, appliances):
        if not isinstance(appliances, list):
            appliances = []

    def update_arcs(self):
        # Undirected edges
        E = []
        for i, neighbors in self.nodes_connected.items():
            for j in neighbors:
                a, b = int(i[1:]), int(j[1:])
                if a != b:
                    E.append((min(a,b), max(a,b)))
        self.E = list(dict.fromkeys(E))

        # Directed arcs
        self.A = []
        for (i,j) in self.E:
            self.A.append((i,j))
            self.A.append((j,i))

    def update_substations(self, w: dict):
        """Used in the deterministic (single-time) model to switch candidate substations 
        to on-line (mark them as existing for the initial constraint).
        Attributes:
        w (dict): Dictionary of substations activation states from previous solution.
        """
        for s, state in w.items():
            # s = [1,2, ...] starts with 1
            if state == 1 and self.SUBSTATIONS[s-1].fix_cost != 0 and self.SUBSTATIONS[s-1].edge_cost != 0:
                self.SUBSTATIONS[s-1].fix_cost = 0
                self.SUBSTATIONS[s-1].edge_cost = 0