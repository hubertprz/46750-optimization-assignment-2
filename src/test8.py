# Working code with a candidate substation implemented classes + add candidate function. Tested on my data
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# ------------------------
#    Classes
# ------------------------
class Substation:
    def __init__(self, id, node, capacity, neighbors, fix_cost = 0, edge_cost = 0):
        self.id = id
        self.node = node
        self.capacity = capacity
        self.neighbors = neighbors
        self.fix_cost = fix_cost    # Fixed cost for activation
        self.edge_cost = edge_cost  # Cost for activating arc connection

class DistributionNetwork:
    def __init__(self, NODES, LOADS, SUBSTATIONS, load_capacity, nodes_connected, loads_locations):
        self.NODES = NODES
        self.LOADS = LOADS
        self.SUBSTATIONS = SUBSTATIONS
        self.load_capacity = load_capacity
        self.nodes_connected = nodes_connected # NEW
        self.loads_locations = loads_locations
        
        # Arcs (distribution lines)
        self.update_arcs()
        self.edge_cost = {e: 0 for e in self.E} # Cost of initial lines set to 0

        self.mapping_substations = pd.DataFrame()
        for S in self.SUBSTATIONS:
            self.mapping_substations.loc[S.id, self.NODES] = 0
            self.mapping_substations.loc[S.id, S.node] = 1

    def add_candidate_substations(self, substations):
        """Adds candidate substations for expansion."""
        # It should:
        # - add substation
        # - add a node [v]
        # - add a node connection [v]
        # - update arcs [v]
        # - set an edge cost [v]
        if not isinstance(substations, list):
            substations = [substations]
        
        for S in substations:
            self.SUBSTATIONS.append(S)
            self.NODES.append(S.node)
            self.nodes_connected[S.node] = S.neighbors
            
            self.update_arcs()
            for neighbor in S.neighbors:
                self.nodes_connected[neighbor].append(S.node) # Also adding connection in nodes that existed previously

                a = int(S.node[1:]) # Number (integer) of the node
                b = int(neighbor[1:]) # Number (integer) of the neighboring node
                e = ((min(a,b), max(a,b))) # Arc (from lower to higher index)
                self.edge_cost[e] = S.edge_cost # Setting edge cost to that arc

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

# ------------------------
# 1️⃣ Define network
# ------------------------
# Important note: Do not define here any nodes associated with candidate substations
NODES = [f"N{i}" for i in range(1,14)] # Listing system nodes: 'N1','N2', ..., 'N13'
LOADS = [f'D{i}' for i in range(1,11)] # Listing system loads: 'D1', 'D2', ..., 'D10'

loads_locations = {
    "D1": "N1",
    "D2": "N2",
    "D3": "N3",
    "D4": "N6",
    "D5": "N7",
    "D6": "N8",
    "D7": "N9",
    "D8": "N11",
    "D9": "N12",
    "D10": "N13",
}
load_capacity = {'D1': 5,
                 'D2': 2,
                 'D3': 2,
                 'D4': 5,
                 'D5': 3,
                 'D6': 2,
                 'D7': 6,
                 'D8': 5,
                 'D9': 3,
                 'D10': 4}

nodes_connected = {
    "N1": ["N2"],
    "N2": ["N1", "N3"],
    "N3": ["N2", "N4"],
    "N4": ["N3", "N5", "N9"],
    "N5": ["N4", "N6"],
    "N6": ["N5","N7", "N8"],
    "N7": ["N6"],
    "N8": ["N6"],
    "N9": ["N4", "N10"],
    "N10": ["N9", "N11", "N13"],
    "N11": ["N10", "N12"],
    "N12": ["N11"],
    "N13": ["N10"]
}

# Existing substation
S1 = Substation("S1", "N4", 100, ["N3", "N5", "N9"])
SUBSTATIONS = [S1]

DistributionNetwork = DistributionNetwork(NODES, LOADS, SUBSTATIONS, load_capacity, nodes_connected, loads_locations)

capacity = 40 # MVA
S2 = Substation("S2", "N14", capacity, ["N2"], fix_cost = 100, edge_cost = 1.0) # Potential substation S2 at node N14, with potential connection to node N2
S3 = Substation("S3", "N15", capacity, ["N5"], fix_cost = 100, edge_cost = 1.0)
S4 = Substation("S4", "N16", capacity, ["N11", "N13"], fix_cost = 100, edge_cost = 1.0)
DistributionNetwork.add_candidate_substations([S2, S3, S4])

# DEBUG
print("Substations", DistributionNetwork.SUBSTATIONS)
print("Nodes", DistributionNetwork.NODES)
print("E",DistributionNetwork.E)
print("A",DistributionNetwork.A)
print("Edge costs", DistributionNetwork.edge_cost)

# ------------------------
# 3️⃣ Network arcs
# ------------------------
# When network is initialized, after arcs are updated, set all edge costs to 0, 
# because those edges already exist (are on-line).
# When you add a substation I want to update all arcs
# and then add those edges to edge_cost with a new price >0.

# ------------------------
# 4️⃣ Model
# ------------------------
def solve_network(Network: DistributionNetwork):
    # ---------------
    #  Index sets
    # ---------------
    N = list(np.arange(1, len(Network.NODES)+1))        # List of node indices
    S = list(np.arange(1, len(Network.SUBSTATIONS)+1))  # List of substations indices

    # Demand vector (demand of each node)
    d = np.zeros(len(Network.NODES))
    for load, node in Network.loads_locations.items():
        ind = int(node[1:]) - 1
        d[ind] = Network.load_capacity[load]

    # Existing substations nodes
    # S_nodes = [int(sub.node[1:]) for sub in Network.SUBSTATIONS if sub.id != "S2"]  # existing only
    S_all_nodes = [int(sub.node[1:]) for sub in Network.SUBSTATIONS]  # all substations

    # Demand/non-substation nodes
    D = np.delete(np.arange(1, len(Network.NODES)+1), np.array(S_all_nodes)-1)  # only nodes without substations

    # ----------------------
    #   Model
    # ----------------------
    model = gp.Model("Radial_Distribution_Network")

    # Parameters
    M = float(np.sum(d)) # Big-M for flow

    # Decision variables
    w = model.addVars(S, vtype=GRB.BINARY, name="w")
    y = model.addVars([(i,s) for i in N for s in S], vtype=GRB.BINARY, name="y")
    x = model.addVars([(i,j,s) for (i,j) in Network.A for s in S], vtype=GRB.BINARY, name="x")
    f = model.addVars([(s,i,j) for s in S for (i,j) in Network.A], lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="f")
    r = model.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    # ------------------------
    # 5️⃣ Constraints
    # ------------------------
    # Power balance
    model.addConstr(quicksum(r[s] for s in S) == np.sum(d), name="power_balance")

    # Node assignment: only demand nodes
    for i in D:
        model.addConstr(quicksum(y[i,s] for s in S) == 1, name=f"assign_{i}")

    for i in N:
        for s in S:
            model.addConstr(y[i,s] <= w[s], name=f"y_le_w_{i}_{s}")

    # Substation node assignment: assigned to self if activated
    for s in S:
        node_idx = int(Network.SUBSTATIONS[s-1].node[1:])
        model.addConstr(y[node_idx,s] == w[s], name=f"substation_assign_{s}")
        # substation nodes cannot be assigned to other substations
        for s2 in S:
            if s2 != s:
                model.addConstr(y[node_idx,s2] == 0, name=f"substation_no_assign_{s}_{s2}")

    # Existing substations must be active (Initial constraint)
    model.addConstr(w[1] == 1, name="w_act_1")

    # Substation supply
    for s in S:
        model.addConstr(r[s] == quicksum(d[i-1]*y[i,s] for i in N), name=f"supply_def_{s}")

    # Capacity
    for s in S:
        model.addConstr(r[s] <= Network.SUBSTATIONS[s-1].capacity * w[s], name=f"capacity_{s}")

    # Flow conservation
    for s in S:
        for i in N:
            incoming = quicksum(f[s,j,i] for (j,k) in Network.A if k==i)
            outgoing = quicksum(f[s,i,j] for (k,j) in Network.A if k==i)
            if i == int(Network.SUBSTATIONS[s-1].node[1:]):  # substation root
                model.addConstr(outgoing - incoming == r[s], name=f"flow_balance_sub_{s}_{i}")
            elif i in D:
                model.addConstr(outgoing - incoming == -d[i-1]*y[i,s], name=f"flow_balance_{s}_{i}")

    # Flow only if arc assigned
    for s in S:
        for (i,j) in Network.A:
            model.addConstr(f[s,i,j] <= M*x[i,j,s], name=f"f_cap_{s}_{i}_{j}")

    # Radiality: each demand node has exactly one parent per assigned substation
    for s in S:
        for i in D:
            model.addConstr(quicksum(x[j,i,s] for (j,k) in Network.A if k==i) == y[i,s], name=f"one_parent_{s}_{i}")
        # substation node has no parent
        node_idx = int(Network.SUBSTATIONS[s-1].node[1:])
        model.addConstr(quicksum(x[j,node_idx,s] for (j,k) in Network.A if k==node_idx) == 0, name=f"parent_root_{s}")

    # Tree size: arcs = nodes assigned - w[s]
    for s in S:
        model.addConstr(quicksum(x[i,j,s] for (i,j) in Network.A) == quicksum(y[k,s] for k in N) - w[s], name=f"tree_size_{s}")

    # Objective: fixed cost + edge cost
    fix_term = quicksum(Network.SUBSTATIONS[s-1].fix_cost * w[s] for s in S)
    edge_term = 0.5*quicksum(Network.edge_cost[(min(i,j),max(i,j))]*x[i,j,s] for (i,j) in Network.A for s in S)
    model.setObjective(fix_term + edge_term, GRB.MINIMIZE)

    # ------------------------
    # 6️⃣ Solve
    # ------------------------
    model.optimize()

    return {
        "model": model,
        "w": w,
        "y": y,
        "x": x,
        "f": f,
        "r": r
    }


# RUN
solution = solve_network(DistributionNetwork)

# ------------------------
# 7️⃣ Results
# ------------------------
def print_results(Network, solution):
    N = list(np.arange(1, len(Network.NODES)+1))  # [1,2,3,4]
    S = list(np.arange(1, len(Network.SUBSTATIONS)+1))  # [1,2]

    print("\nSubstation activation and supply:")
    for s in S:
        print(f"{Network.SUBSTATIONS[s-1].id} at {Network.SUBSTATIONS[s-1].node}: w={solution['w'][s].X}, r={solution['r'][s].X}")

    print("\nNode assignments:")
    for i in N:
        for s in S:
            if solution['y'][i,s].X > 0.5:
                print(f"Node {Network.NODES[i-1]} assigned to {Network.SUBSTATIONS[s-1].id}")

    print("\nArcs used:")
    for (i,j,s) in solution['x'].keys():
        if solution['x'][i,j,s].X > 0.5:
            print(f"Arc {Network.NODES[i-1]} -> {Network.NODES[j-1]} assigned to {Network.SUBSTATIONS[s-1].id}")

print_results(DistributionNetwork, solution)
