import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

from src.classes import DistributionNetwork

def solve_network(Network: DistributionNetwork, OutputFlag=0):
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

    # Substations nodes
    S_all_nodes = [int(sub.node[1:]) for sub in Network.SUBSTATIONS]  # all substations nodes

    # Demand/non-substation nodes
    D = np.delete(np.arange(1, len(Network.NODES)+1), np.array(S_all_nodes)-1)  # only nodes without substations

    # ----------------------
    #   Model
    # ----------------------
    model = gp.Model("Radial_Distribution_Network")

    #   Model. Parameters
    M = float(np.sum(d)) # Big-M for flow

    #   Model. Decision variables
    w = model.addVars(S, vtype=GRB.BINARY, name="w")
    y = model.addVars([(i,s) for i in N for s in S], vtype=GRB.BINARY, name="y")
    x = model.addVars([(i,j,s) for (i,j) in Network.A for s in S], vtype=GRB.BINARY, name="x")
    f = model.addVars([(s,i,j) for s in S for (i,j) in Network.A], lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="f")
    r = model.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    #   Model. Constraints
    #
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
    # Looks for substations with ZERO FIXED COST.
    for s in S:
        if Network.SUBSTATIONS[s-1].fix_cost == 0:
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

    #   Model. Objective function
    # 
    #   f(x) = fixed cost + edge cost
    fix_term = quicksum(Network.SUBSTATIONS[s-1].fix_cost * w[s] for s in S)
    edge_term = 0.5*quicksum(Network.edge_cost[(min(i,j),max(i,j))]*x[i,j,s] for (i,j) in Network.A for s in S)
    model.setObjective(fix_term + edge_term, GRB.MINIMIZE)

    # ------------------------
    #   Solve
    # ------------------------
    model.setParam('OutputFlag', OutputFlag) # Decide if to display the optimization output (Def. = 0)
    model.optimize()

    # Extract numerical results
    w_val = {s: w[s].X for s in S}
    y_val = {(i, s): y[i, s].X for i in N for s in S}
    x_val = {(i, j, s): x[i, j, s].X for (i, j) in Network.A for s in S}
    f_val = {(s, i, j): f[s, i, j].X for s in S for (i, j) in Network.A}
    r_val = {s: r[s].X for s in S}

    # Return numerical data
    return {
        "objective": model.ObjVal,
        "w": w_val,
        "y": y_val,
        "x": x_val,
        "f": f_val,
        "r": r_val
    }


def print_results(Network, solution):
    N = list(np.arange(1, len(Network.NODES)+1))  # [1,2,3,4]
    S = list(np.arange(1, len(Network.SUBSTATIONS)+1))  # [1,2]

    print("\nSubstation activation and supply:")
    for s in S:
        print(f"{Network.SUBSTATIONS[s-1].id} at {Network.SUBSTATIONS[s-1].node}: w={solution['w'][s]}, r={solution['r'][s]}")

    print("\nNode assignments:")
    for i in N:
        for s in S:
            if solution['y'][i,s] > 0.5:
                print(f"Node {Network.NODES[i-1]} assigned to {Network.SUBSTATIONS[s-1].id}")

    print("\nArcs used:")
    for (i,j,s) in solution['x'].keys():
        if solution['x'][i,j,s] > 0.5:
            print(f"Arc {Network.NODES[i-1]} -> {Network.NODES[j-1]} assigned to {Network.SUBSTATIONS[s-1].id}")