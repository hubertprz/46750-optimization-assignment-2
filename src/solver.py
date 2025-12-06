import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

from src.classes import DistributionNetwork

def solve_network(Network: DistributionNetwork, R, B, opex_cost, dr, Y, OutputFlag=0):
    """Solves a Distribution Network expansion problem.

    Args:
        Network (DistributionNetwork): Defined Distribution Network.
        opex_cost (float): Annual operational cost per substation.
        R (float): Size of a single capacity reinforcement.
        B (float): Annual budget (nominal value).
        dr (float): Discount rate.
        Y (int): Year (ONLY FOR MODEL 1). Used for cost discounting.
        OutputFlag (int, optional): Display solver output. Defaults to 0.

    Returns:
        dict: Dictionary with values of objective function, decision variables and system
              capacities.
    """
    #   Sets
    N = list(np.arange(1, len(Network.NODES)+1))                # List of node indices starting from 1 [1, 2, ...]
    S = list(np.arange(1, len(Network.SUBSTATIONS)+1))          # List of substations indices starting from 1 [1, 2, ...]
    S_0 = [int(s.id[1:]) for s in Network.SUBSTATIONS if s.fix_cost == 0] # Set of existing (online) substations (for initial condition)

    A = Network.A                                               # Set of arcs (i,j)
    N_S = [int(sub.node[1:]) for sub in Network.SUBSTATIONS]    # Set of (all) substation nodes
    N_NS = np.delete(np.array(N), np.array(N_S)-1)              # Set of non-substation nodes (=N \ N_S)

    # Demand vector (demand of each node)
    L = Network.load_capacity                                   # Power consumption (demand) of loads
    d = np.zeros(len(Network.NODES))
    for load, node in Network.loads_locations.items():
        ind = int(node[1:]) - 1
        d[ind] = L[load]
    # d vector is identical to L mapped onto the node set. Preferred version for computations

    # ----------------------
    #   Model
    # ----------------------
    model = gp.Model("Radial_Distribution_Network")

    #   Model. Parameters
    # L = Network.load_capacity (above)                 # Demand of loads. Mapped onto the node set (d vector) is preferred.
    P = [s.capacity for s in Network.SUBSTATIONS]       # Capacity of substation s
    C_S = [s.fix_cost for s in Network.SUBSTATIONS]     # Cost of activating substation s
    C_L = Network.edge_cost                             # Cost of connecting line (i,j)
    C_R = [s.r_cost for s in Network.SUBSTATIONS]       # Cost of capacity reinforcement
    C_OPEX = opex_cost
    # R from the function arguments                     # Size of a single capacity reinforcement
    # B ...                                             # Annual budget
    # dr ...                                            # Discount rate
    # Y ...                                             # Year (only in single-period Model 1)
    M = float(np.sum(d))                                # Big-M used to constraint power flow

    #   Model. Decision variables
    w = model.addVars(S, vtype=GRB.BINARY, name="w") # Substation Activation
    y = model.addVars([(i,s) for i in N for s in S], vtype=GRB.BINARY, name="y") # Node Assignment (to substation)
    x = model.addVars([(i,j,s) for (i,j) in Network.A for s in S], vtype=GRB.BINARY, name="x") # Arc Usage (by substation s)
    z = model.addVars(S, lb=0, vtype=GRB.INTEGER, name="z") # Capacity Reinforcement
    f = model.addVars([(s,i,j) for s in S for (i,j) in Network.A], lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="f") # Power Flow
    r = model.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="r") # Substation Supply

    #   Model. Constraints
    #   (1) Power balance
    #   All power flows sum up to total system demand (100% covered).
    model.addConstr(quicksum(r[s] for s in S) == np.sum(d), name="power_balance")

    #   (2 ) Node Assignment
    #   Node assigned to a substation y[n,s] = 1 only if substation is active w[s] = 1.
    for n in N:
        for s in S:
            model.addConstr(y[n,s] <= w[s], name=f"y_le_w_{n}_{s}")

    #   (2a) Non-substation nodes
    #   Non-substation node must be assigned to one, and only one, substation.
    for n in N_NS:
        model.addConstr(quicksum(y[n,s] for s in S) == 1, name=f"assign_{n}")

    #   (2b) Substation nodes
    #   Substation node assignment: assigned to self (if activated).
    for s in S:
        node_idx = N_S[s-1]
        model.addConstr(y[node_idx,s] == w[s], name=f"substation_assign_{s}")
        # Substation nodes cannot be assigned to other substations.
        for s2 in S:
            if s2 != s:
                model.addConstr(y[node_idx,s2] == 0, name=f"substation_no_assign_{s}_{s2}")

    #   (3) Substation supply
    #   Power flow from a substation is equal to demands supplied by this substation.
    for s in S:
        model.addConstr(r[s] == quicksum(d[n-1]*y[n,s] for n in N), name=f"supply_def_{s}")

    #   (4) Capacity constraint
    #   Power flow from a substation lower or equal to substation's max capacity + potential capacity reinforcements.
    for s in S:
        model.addConstr(r[s] <= (P[s-1] + R * z[s]) * w[s], name=f"capacity_{s}")

    #   (5 ) Substation Feeder Line
    #   (5a) Activation - optional, i think it makes sense
    #   If a substation is activated, at least one feeder line is always used by flows from it.
    #   In practice: You can't fully disconnect a substation once it is activated.
    #   Now activating a substation can constraint future developments, so multi-period
    #   optimization becomes more valuable.
    for s in S:
        model.addConstr(w[s] <= quicksum(x[i,j,s] for (i,j) in A))           

    #   (6) Flow conservation
    for s in S:
        for n in N:
            incoming = quicksum(f[s,j,n] for (j,k) in A if k==n)
            outgoing = quicksum(f[s,n,j] for (k,j) in A if k==n)
            if n == N_S[s-1]:  # Substation root (if node is a substation)
                model.addConstr(outgoing - incoming == r[s], name=f"flow_balance_sub_{s}_{n}")
            elif n in N_NS:
                model.addConstr(outgoing - incoming == -d[n-1]*y[n,s], name=f"flow_balance_{s}_{n}")

    #   (7) Flow only if arc assigned
    #   Flow through arc is costrained by total system demand. Equal to zero if arc is inactive.
    for s in S:
        for (i,j) in A:
            model.addConstr(f[s,i,j] <= M*x[i,j,s], name=f"f_cap_{s}_{i}_{j}")

    #   (8) Radiality
    #   Each non-substation node has exactly one parent per assigned substation (1).
    for s in S:
        for n in N_NS:
            model.addConstr(quicksum(x[j,n,s] for (j,k) in A if k==n) == y[n,s], name=f"one_parent_{s}_{n}")

        # Substation node has no parent (no flows INTO substation node) (2).
        node_idx = N_S[s-1]
        model.addConstr(quicksum(x[j,node_idx,s] for (j,k) in A if k==node_idx) == 0, name=f"parent_root_{s}")

        # Substation node n_s assigned to the substation s forbids other substations from using outgoing arcs from n_s (3).
        for (k, j) in A:
            if k == node_idx:
                model.addConstr(
                    quicksum(x[k, j, s2] for s2 in S if s2 != s) <= (1 - y[node_idx, s]),
                    name=f"root_arc_ass_s{s}_arc{node_idx}_{j}")

    #   (9) Tree size
    for s in S:
        model.addConstr(quicksum(x[i,j,s] for (i,j) in A) == quicksum(y[n,s] for n in N) - w[s], name=f"tree_size_{s}")

    #   (10 ) Initial Constraints
    #   Existing substations activation
    #   Existing substations must stay active. (Looks for substations with ZERO FIXED COST)
    for s in S_0:
        model.addConstr(w[s] == 1, name="w_act_{s}")

    #   (11) Annual budget costraint
    #   Total cost cannot exceed annual budget.
    fix_term = quicksum(C_S[s-1] * w[s] for s in S)
    edge_term = quicksum(C_L[(min(i,j),max(i,j))] * x[i,j,s] for (i,j) in A for s in S)
    reinf_term = quicksum(C_R[s-1] * z[s] for s in S)
    opex_term = quicksum(w[s] * C_OPEX for s in S)
    model.addConstr(fix_term + edge_term + reinf_term + opex_term <= B, name=f"budget_{Y}") # Nominal prices

    #   Model. Objective function
    #   min f(x) = substation cost + feeder cost + capacity reinforcement cost + OPEX

    model.setObjective((fix_term + edge_term + reinf_term + opex_term)/(1+dr)**(Y-1), GRB.MINIMIZE)

    # ------------------------
    #   Solve
    # ------------------------
    model.setParam('OutputFlag', OutputFlag) # Decide if to display the optimization output (Def. = 0)
    model.optimize()

    # Extract numerical results (if feasible)
    if model.status == GRB.INFEASIBLE:
        raise Exception("Model is infeasible. Check budget constraints.")
    else:
        cost_val = (sum(C_S[s-1] * w[s].X for s in S) 
                    + sum(C_L[(min(i,j), max(i,j))] * x[i,j,s].X for (i,j) in A for s in S)
                    + sum(C_R[s-1] * z[s].X for s in S)
                    + sum (w[s].X * C_OPEX for s in S))
        w_val = {s: w[s].X for s in S}
        y_val = {(i, s): y[i, s].X for i in N for s in S}
        x_val = {(i, j, s): x[i, j, s].X for (i, j) in A for s in S}
        f_val = {(s, i, j): f[s, i, j].X for s in S for (i, j) in A}
        r_val = {s: r[s].X for s in S}
        z_val = {s: z[s].X for s in S}
        P_val = {s: (P[s-1] + z[s].X * R) * w[s].X for s in S} # Capacities

    # Return numerical data
    return {
            "objective": model.ObjVal,  # Discounted cost
            "cost": cost_val,
            "w": w_val,
            "y": y_val,
            "x": x_val,
            "f": f_val,
            "r": r_val,
            "z": z_val,
            "P": P_val
            }


def print_results(Network, solution, detailed=False):
    N = list(np.arange(1, len(Network.NODES)+1))  # [1,2,3,4]
    S = list(np.arange(1, len(Network.SUBSTATIONS)+1))  # [1,2]
    print("System demand:", round(sum(Network.load_capacity.values()),2))
    print("Total cost (nominal):", solution['cost'])
    print("Total discounted cost:", round(solution['objective'], 2))

    print("\nSubstation activation and supply:")
    for s in S:
        print(f"{Network.SUBSTATIONS[s-1].id} at {Network.SUBSTATIONS[s-1].node}: w={solution['w'][s]}, r={solution['r'][s]:.2f}")

    if detailed:
        print("\nNode assignments:")
        for i in N:
            for s in S:
                if solution['y'][i,s] > 0.5:
                    print(f"Node {Network.NODES[i-1]} assigned to {Network.SUBSTATIONS[s-1].id}")

        print("\nArcs used:")
        for (i,j,s) in solution['x'].keys():
            if solution['x'][i,j,s] > 0.5:
                print(f"Arc {Network.NODES[i-1]} -> {Network.NODES[j-1]} assigned to {Network.SUBSTATIONS[s-1].id}")