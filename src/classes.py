import pandas as pd
from math import isclose

class Substation:
    def __init__(self, id, node, capacity, neighbors, fix_cost = 0, edge_cost = 0):
        self.id = id
        self.node = node
        self.capacity = capacity
        self.neighbors = neighbors  # Nodes with potential connection
        self.fix_cost = fix_cost    # Fixed cost for activation
        self.edge_cost = edge_cost  # Cost for activating arc connection (feeder lines)

class Heatpump():
    """Potential class to utilize"""
    def __init__(self, id, node, load):
        self.id = id
        self.node = node
        self.load = load

class EV():
    """Potential class to utilize"""
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
        nodes_connected (dict):
        loads_locations (dict):
        line_cost (float): Default cost of building network lines (different from substation feeder lines)

    Methods:
        add_candidate_substations:
        update_arcs:
        update_initial_conditions:
    """
    def __init__(self, NODES, LOADS, SUBSTATIONS, load_capacity, nodes_connected, loads_locations, line_cost):
        self.NODES = NODES
        self.LOADS = LOADS
        self.SUBSTATIONS = SUBSTATIONS
        self.load_capacity = load_capacity
        self.nodes_connected = nodes_connected
        self.loads_locations = loads_locations
        self.line_cost = line_cost  # 

        # Arcs (distribution lines)
        self.update_arcs()
        self.edge_cost = {e: 0 for e in self.E} # Initial condition for "existing lines". Cost of initial lines set to 0
        self.A_activated = [] # *Only for MODEL 1: List of activated distribution lines from previous time periods 

        self.mapping_substations = pd.DataFrame()
        for S in self.SUBSTATIONS:
            self.mapping_substations.loc[S.id, self.NODES] = 0
            self.mapping_substations.loc[S.id, S.node] = 1


    def add_candidate_substations(self, substations):
        """Adds candidate substations for expansion. 
        Substations can't be connected to each other.
        """
        if not isinstance(substations, list):
            substations = [substations]
        
        for S in substations:
            self.SUBSTATIONS.append(S)
            self.NODES.append(S.node)
            self.nodes_connected[S.node] = S.neighbors        # Adding connections to neighbors in the new node
            
            self.update_arcs()
            for neighbor in S.neighbors:
                self.nodes_connected[neighbor].append(S.node) # Also adding connection in neighboring nodes that existed before

                a = int(S.node[1:])             # Index (integer) of the node
                b = int(neighbor[1:])           # Index (integer) of the neighboring node
                e = ((min(a,b), max(a,b)))      # Arc (from lower to higher index)
                self.edge_cost[e] = S.edge_cost # Setting edge cost to that arc

    def add_appliances(self, appliances):
        """Method suggestion for future models."""
        if not isinstance(appliances, list):
            appliances = [appliances]

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

    def update_initial_conditions(self, w: dict, x: dict, z: dict):
        """
        Used in Model 1 (single-period optimization).

        Attributes:
            w (dict): Dictionary of substations activation variables from previous solution.
        """
        # --- Substations ---
        # Mark activated substations from a previous time period as on-line in the next time period. 
        for s, state in w.items():
            # s = [1,2, ...] starts with 1
            if state == 1 and self.SUBSTATIONS[s-1].fix_cost != 0 and self.SUBSTATIONS[s-1].edge_cost != 0:
                self.SUBSTATIONS[s-1].fix_cost = 0
                self.SUBSTATIONS[s-1].edge_cost = 0

        # --- Distribution lines ---
        # Every line coming from a newly activated substation will now always be active 
        # (aka you cant disconnect a built substation)
        for (i,j,s), state in x.items():
            e = ((min(i,j), max(i,j)))
            if state == 1 and self.edge_cost[e] != 0:
                self.edge_cost[e] = 0                           # Changes edge cost to 0
                self.A_activated.append((i,j,s))                # Adds edge to A_activated so its always active from now

        # Every inactive line, should now have a cost.
        # After first initialization all "existing" lines had 0 cost, so they could be freely build by a solver.
        # But after first year, if there are any inactive lines among those that were initially existing,
        # meaning there are gaps now (due to new substations active in the network), REBUILDING those lines
        # should cost. This is done so the solver does not freely jump assigning different paths ("portioning demands")
        # each time without any cost.

        for (i,j) in self.E:
            state_sum = 0
            for s in range(1,len(self.SUBSTATIONS)+1,1):
                state = x[(i,j,s)] + x[(j,i,s)] # Look for disactivation of an arc in any direction (i,j) or (j,i)
                state_sum += state
            
            if isclose(state_sum, 0) and self.edge_cost[(i,j)] == 0:
                print("Arc disconnected:", (i,j))
                self.edge_cost[(i,j)] = self.line_cost

        # Reinforced existing substations (increased capacity)
        for s, multiplier in z.items():
            self.SUBSTATIONS[s-1].capacity += multiplier * 10   # adds (multiplier)*10 to capacity