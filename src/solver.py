import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

from src.classes import DistributionNetwork

def solve_network(Network: DistributionNetwork, R, B, OutputFlag=0):
    """Solves a Distribution Network expansion problem.

    Args:
        Network (DistributionNetwork): Defined Distribution Network.
        R (float): Size of a single capacity reinforcement.
        B (float): Budget (nominal value).
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
    # R from the function arguments                     # Size of a single capacity reinforcement
    # B ...                                             # Annual budget
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
    #   All power flows sum up to total system demand (100% covered)
    model.addConstr(quicksum(r[s] for s in S) == np.sum(d), name="power_balance")

    #   (2 ) Node Assignment
    #   Node assigned to a substation y[n,s] = 1 only if substation is active w[s] = 1
    for n in N:
        for s in S:
            model.addConstr(y[n,s] <= w[s], name=f"y_le_w_{n}_{s}")

    #   (2a) Non-substation nodes
    #   Non-substation node must be assigned to one, and only one, substation
    for n in N_NS:
        model.addConstr(quicksum(y[n,s] for s in S) == 1, name=f"assign_{n}")

    #   (2b) Substation nodes
    #   Substation node assignment: assigned to self (if activated)
    for s in S:
        node_idx = N_S[s-1]
        model.addConstr(y[node_idx,s] == w[s], name=f"substation_assign_{s}")
        # Substation nodes cannot be assigned to other substations
        for s2 in S:
            if s2 != s:
                model.addConstr(y[node_idx,s2] == 0, name=f"substation_no_assign_{s}_{s2}")

    #   (3) Substation supply
    #   Power flow from a substation is equal to demands supplied by this substation
    for s in S:
        model.addConstr(r[s] == quicksum(d[n-1]*y[n,s] for n in N), name=f"supply_def_{s}")

    #   (4) Capacity constraint
    #   Power flow from a substation lower or equal to substation's max capacity + potential capacity reinforcements
    for s in S:
        model.addConstr(r[s] <= (P[s-1] + R * z[s]) * w[s], name=f"capacity_{s}")

    #   (5 ) Substation Feeder Line
    #   (5a) Activation - optional, i think it makes sense
    #   If a substation is activated, at least one feeder line is always used by flows from it
    #   In practice: You can't fully disconnect a substation once it is activated
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
    #   Flow through arc is costrained by total system demand. Equal to zero if arc is inactive 
    for s in S:
        for (i,j) in A:
            model.addConstr(f[s,i,j] <= M*x[i,j,s], name=f"f_cap_{s}_{i}_{j}")

    #   (8) Radiality
    #   Each non-substation node has exactly one parent per assigned substation (1)
    for s in S:
        for n in N_NS:
            model.addConstr(quicksum(x[j,n,s] for (j,k) in A if k==n) == y[n,s], name=f"one_parent_{s}_{n}")

        # Substation node has no parent (no flows INTO substation node) (2)
        node_idx = N_S[s-1]
        model.addConstr(quicksum(x[j,node_idx,s] for (j,k) in A if k==node_idx) == 0, name=f"parent_root_{s}")

        # Substation node n_s assigned to the substation s forbids other substations from using outgoing arcs from n_s (3)
        for (k, j) in A:
            if k == node_idx:
                model.addConstr(
                    quicksum(x[k, j, s2] for s2 in S if s2 != s) <= (1 - y[node_idx, s]),
                    name=f"root_arc_ass_s{s}_arc{node_idx}_{j}")

    #   (9) Tree size: arcs = nodes assigned - w[s]
    for s in S:
        model.addConstr(quicksum(x[i,j,s] for (i,j) in A) == quicksum(y[n,s] for n in N) - w[s], name=f"tree_size_{s}")

    #   (10 ) Initial Constraints
    #   Existing substations activation
    #   EXISTING substations must stay active. (Looks for substations with ZERO FIXED COST)
    for s in S_0:
        model.addConstr(w[s] == 1, name="w_act_{s}")

    #   (11) Annual budget costraint
    #   Total cost cannot exceed annual budget
    fix_term = quicksum(C_S[s-1] * w[s] for s in S)
    edge_term = quicksum(C_L[(min(i,j),max(i,j))] * x[i,j,s] for (i,j) in A for s in S)
    reinf_term = quicksum(C_R[s-1] * z[s] for s in S)
    model.addConstr(fix_term + edge_term + reinf_term <= B, name=f"budget") # Nominal prices

    #   Model. Objective function
    #   min f(x) = substation cost + feeder cost + capacity reinforcement cost

    model.setObjective(fix_term + edge_term + reinf_term, GRB.MINIMIZE)

    # ------------------------
    #   Solve
    # ------------------------
    model.setParam('OutputFlag', OutputFlag) # Decide if to display the optimization output (Def. = 0)
    model.optimize()

    # Extract numerical results (if feasible)
    if model.status == GRB.INFEASIBLE:
        raise Exception("Model is infeasible. Check budget constraints.")
    else:
        w_val = {s: w[s].X for s in S}
        y_val = {(i, s): y[i, s].X for i in N for s in S}
        x_val = {(i, j, s): x[i, j, s].X for (i, j) in A for s in S}
        f_val = {(s, i, j): f[s, i, j].X for s in S for (i, j) in A}
        r_val = {s: r[s].X for s in S}
        z_val = {s: z[s].X for s in S}
        P_val = {s: (P[s-1] + z[s].X * R) * w[s].X for s in S} # Capacities

    # Return numerical data
    return {
            "objective": model.ObjVal,  # Cost
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
    print("Total cost (nominal):", round(solution['objective'], 2))

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


# ------------------------------------------------------------
# Intertemporal (multi-year) extension of the network problem
# ------------------------------------------------------------
def solve_network_intertemporal(Network: DistributionNetwork, R, B, dr, years, demands, op_cost=0.0,
                                 connect_cost=25.0, disconnect_cost=25.0, OutputFlag=0):
    """Multi-year deterministic expansion model (Model 2).

    Investments are shared across years (once-built stays built). Budgets include
    substation activations, line builds, reinforcements, opex, and connection changes.
    """
    # Sets
    N = list(np.arange(1, len(Network.NODES)+1))
    S = list(np.arange(1, len(Network.SUBSTATIONS)+1))
    S_0 = [int(s.id[1:]) for s in Network.SUBSTATIONS if s.fix_cost == 0]
    A = Network.A
    E = Network.E
    N_S = [int(sub.node[1:]) for sub in Network.SUBSTATIONS]
    N_NS = np.delete(np.array(N), np.array(N_S)-1)

    # Parameters
    P = [s.capacity for s in Network.SUBSTATIONS]
    C_S = [s.fix_cost for s in Network.SUBSTATIONS]
    C_L = Network.edge_cost
    C_R = [s.r_cost for s in Network.SUBSTATIONS]

    # Demand per year mapped to nodes
    d_t = {}
    for t in years:
        L = demands[t]
        d = np.zeros(len(Network.NODES))
        for load, node in Network.loads_locations.items():
            ind = int(node[1:]) - 1
            d[ind] = L[load]
        d_t[t] = d

    model = gp.Model("Radial_Distribution_Network_Intertemporal")

    # Investment & operation variables
    w = model.addVars(S, years, vtype=GRB.BINARY, name="w")
    w_on = model.addVars(S, years, vtype=GRB.BINARY, name="w_on")
    b = model.addVars([(i, j, s, t) for (i, j) in E for s in S for t in years], vtype=GRB.BINARY, name="b")
    b_on = model.addVars([(i, j, s, t) for (i, j) in E for s in S for t in years], vtype=GRB.BINARY, name="b_on")
    y = model.addVars([(i, s, t) for i in N for s in S for t in years], vtype=GRB.BINARY, name="y")
    x = model.addVars([(i, j, s, t) for (i, j) in A for s in S for t in years], vtype=GRB.BINARY, name="x")
    x_on = model.addVars([(i, j, s, t) for (i, j) in A for s in S for t in years], vtype=GRB.BINARY, name="x_on")
    x_off = model.addVars([(i, j, s, t) for (i, j) in A for s in S for t in years], vtype=GRB.BINARY, name="x_off")
    z = model.addVars(S, years, lb=0, vtype=GRB.INTEGER, name="z")
    f = model.addVars([(s, i, j, t) for s in S for (i, j) in A for t in years], lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    r = model.addVars(S, years, lb=0.0, vtype=GRB.CONTINUOUS, name="r")

    # Monotone activations
    for s in S:
        first_t = years[0]
        model.addConstr(w_on[s, first_t] == w[s, first_t], name=f"w_on_init_{s}")
        if s in S_0:
            model.addConstr(w[s, first_t] == 1, name=f"w0_fix_{s}")
        for idx in range(1, len(years)):
            t_prev = years[idx-1]
            t_cur = years[idx]
            model.addConstr(w[s, t_cur] >= w[s, t_prev], name=f"w_monotone_{s}_{t_cur}")
            model.addConstr(w_on[s, t_cur] >= w[s, t_cur] - w[s, t_prev], name=f"w_on_diff_{s}_{t_cur}")
            if s in S_0:
                model.addConstr(w[s, t_cur] == 1, name=f"w_fix_{s}_{t_cur}")

    # Monotone line builds
    for s in S:
        for (i_u, j_u) in E:
            first_t = years[0]
            model.addConstr(b_on[i_u, j_u, s, first_t] == b[i_u, j_u, s, first_t], name=f"b_on_init_{s}_{i_u}_{j_u}")
            for idx in range(1, len(years)):
                t_prev = years[idx-1]
                t_cur = years[idx]
                model.addConstr(b[i_u, j_u, s, t_cur] >= b[i_u, j_u, s, t_prev], name=f"b_monotone_{s}_{i_u}_{j_u}_{t_cur}")
                model.addConstr(b_on[i_u, j_u, s, t_cur] >= b[i_u, j_u, s, t_cur] - b[i_u, j_u, s, t_prev],
                                name=f"b_on_diff_{s}_{i_u}_{j_u}_{t_cur}")

    # Constraints by year
    for t in years:
        d = d_t[t]
        M = float(np.sum(d))

        model.addConstr(quicksum(r[s, t] for s in S) == np.sum(d), name=f"power_balance_{t}")

        for n in N:
            for s in S:
                model.addConstr(y[n, s, t] <= w[s, t], name=f"y_le_w_{n}_{s}_{t}")

        for n in N_NS:
            model.addConstr(quicksum(y[n, s, t] for s in S) == 1, name=f"assign_{n}_{t}")

        for s in S:
            node_idx = N_S[s-1]
            model.addConstr(y[node_idx, s, t] == w[s, t], name=f"substation_assign_{s}_{t}")
            for s2 in S:
                if s2 != s:
                    model.addConstr(y[node_idx, s2, t] == 0, name=f"substation_no_assign_{s}_{s2}_{t}")

        for s in S:
            model.addConstr(r[s, t] == quicksum(d[n-1] * y[n, s, t] for n in N), name=f"supply_def_{s}_{t}")

        for s in S:
            cap = P[s-1] + R * quicksum(z[s, tau] for tau in years if tau <= t)
            model.addConstr(r[s, t] <= cap * w[s, t], name=f"capacity_{s}_{t}")

        for s in S:
            model.addConstr(w[s, t] <= quicksum(x[i, j, s, t] for (i, j) in A), name=f"feeder_{s}_{t}")

        for s in S:
            for n in N:
                incoming = quicksum(f[s, j, n, t] for (j, k) in A if k == n)
                outgoing = quicksum(f[s, n, j, t] for (k, j) in A if k == n)
                if n == N_S[s-1]:
                    model.addConstr(outgoing - incoming == r[s, t], name=f"flow_balance_sub_{s}_{n}_{t}")
                elif n in N_NS:
                    model.addConstr(outgoing - incoming == -d[n-1] * y[n, s, t], name=f"flow_balance_{s}_{n}_{t}")

        for s in S:
            for (i, j) in A:
                model.addConstr(f[s, i, j, t] <= M * x[i, j, s, t], name=f"f_cap_{s}_{i}_{j}_{t}")
                e = (min(i, j), max(i, j))
                model.addConstr(x[i, j, s, t] <= b[e[0], e[1], s, t], name=f"x_le_b_{s}_{i}_{j}_{t}")
                # Connection change tracking
                if t == years[0]:
                    # No connection/disconnection cost in first year
                    model.addConstr(x_on[i, j, s, t] == 0, name=f"x_on_init_{s}_{i}_{j}_{t}")
                    model.addConstr(x_off[i, j, s, t] == 0, name=f"x_off_init_{s}_{i}_{j}_{t}")
                else:
                    t_prev = years[years.index(t)-1]
                    model.addConstr(x_on[i, j, s, t] >= x[i, j, s, t] - x[i, j, s, t_prev], name=f"x_on_diff_{s}_{i}_{j}_{t}")
                    model.addConstr(x_on[i, j, s, t] <= x[i, j, s, t], name=f"x_on_le_x_{s}_{i}_{j}_{t}")
                    model.addConstr(x_off[i, j, s, t] >= x[i, j, s, t_prev] - x[i, j, s, t], name=f"x_off_diff_{s}_{i}_{j}_{t}")
                    model.addConstr(x_off[i, j, s, t] <= x[i, j, s, t_prev], name=f"x_off_le_prev_{s}_{i}_{j}_{t}")

        for s in S:
            for n in N_NS:
                model.addConstr(quicksum(x[j, n, s, t] for (j, k) in A if k == n) == y[n, s, t], name=f"one_parent_{s}_{n}_{t}")
            node_idx = N_S[s-1]
            model.addConstr(quicksum(x[j, node_idx, s, t] for (j, k) in A if k == node_idx) == 0, name=f"parent_root_{s}_{t}")
            for (k, j) in A:
                if k == node_idx:
                    model.addConstr(quicksum(x[k, j, s2, t] for s2 in S if s2 != s) <= (1 - y[node_idx, s, t]),
                                    name=f"root_arc_ass_s{s}_arc{node_idx}_{j}_{t}")
        for s in S:
            model.addConstr(quicksum(x[i, j, s, t] for (i, j) in A) == quicksum(y[n, s, t] for n in N) - w[s, t],
                            name=f"tree_size_{s}_{t}")

        # Budget per year (nominal)
        budget_t = B[t] if isinstance(B, dict) else B
        fix_term_t = quicksum(C_S[s-1] * w_on[s, t] for s in S)
        edge_term_t = quicksum(C_L[(i, j)] * b_on[i, j, s, t] for (i, j) in E for s in S)
        connect_term_t = connect_cost * quicksum(x_on[i, j, s, t] for (i, j) in A for s in S)
        disconnect_term_t = disconnect_cost * quicksum(x_off[i, j, s, t] for (i, j) in A for s in S)
        reinf_term_t = quicksum(C_R[s-1] * z[s, t] for s in S)
        opex_term_t = op_cost * quicksum(w[s, t] for s in S)
        model.addConstr(fix_term_t + edge_term_t + connect_term_t + disconnect_term_t + reinf_term_t + opex_term_t <= budget_t, name=f"budget_{t}")

    # Objective: discounted cost
    objective = quicksum(
        (quicksum(C_S[s-1] * w_on[s, t] for s in S) +
         quicksum(C_L[(i, j)] * b_on[i, j, s, t] for (i, j) in E for s in S) +
         connect_cost * quicksum(x_on[i, j, s, t] for (i, j) in A for s in S) +
         disconnect_cost * quicksum(x_off[i, j, s, t] for (i, j) in A for s in S) +
         quicksum(C_R[s-1] * z[s, t] for s in S) +
         op_cost * quicksum(w[s, t] for s in S)) / ((1 + dr) ** (t - 1))
        for t in years
    )
    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam('OutputFlag', OutputFlag)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        raise Exception("Intertemporal model is infeasible. Check budget constraints or demands.")

    # Extract results
    w_val = {(s, t): w[s, t].X for s in S for t in years}
    w_on_val = {(s, t): w_on[s, t].X for s in S for t in years}
    y_val = {(i, s, t): y[i, s, t].X for i in N for s in S for t in years}
    x_val = {(i, j, s, t): x[i, j, s, t].X for (i, j) in A for s in S for t in years}
    f_val = {(s, i, j, t): f[s, i, j, t].X for s in S for (i, j) in A for t in years}
    r_val = {(s, t): r[s, t].X for s in S for t in years}
    z_val = {(s, t): z[s, t].X for s in S for t in years}
    P_val = {(s, t): (P[s-1] + R * sum(z[s, tau].X for tau in years if tau <= t)) * w_val[s, t]
             for s in S for t in years}
    b_val = {(i, j, s, t): b[i, j, s, t].X for (i, j) in E for s in S for t in years}
    b_on_val = {(i, j, s, t): b_on[i, j, s, t].X for (i, j) in E for s in S for t in years}
    x_on_val = {(i, j, s, t): x_on[i, j, s, t].X for (i, j) in A for s in S for t in years}
    x_off_val = {(i, j, s, t): x_off[i, j, s, t].X for (i, j) in A for s in S for t in years}

    cost_per_year = {}
    cost_per_year_nominal = {}
    cost_components_nominal = {}
    cost_components_discounted = {}
    for t in years:
        fix_term_t = sum(C_S[s-1] * w_on_val[s, t] for s in S)
        edge_term_t = sum(C_L[(i, j)] * b_on_val[i, j, s, t] for (i, j) in E for s in S)
        connect_term_t = connect_cost * sum(x_on_val[i, j, s, t] for (i, j) in A for s in S)
        disconnect_term_t = disconnect_cost * sum(x_off_val[i, j, s, t] for (i, j) in A for s in S)
        reinf_term_t = sum(C_R[s-1] * z_val[s, t] for s in S)
        opex_t = op_cost * sum(w_val[s, t] for s in S)
        cost_per_year_nominal[t] = fix_term_t + edge_term_t + connect_term_t + disconnect_term_t + reinf_term_t + opex_t
        cost_per_year[t] = cost_per_year_nominal[t] / ((1 + dr) ** (t - 1))
        cost_components_nominal[t] = {
            "substation": fix_term_t,
            "lines": edge_term_t,
            "connect": connect_term_t,
            "disconnect": disconnect_term_t,
            "reinforcement": reinf_term_t,
            "opex": opex_t,
        }
        cost_components_discounted[t] = {
            k: v / ((1 + dr) ** (t - 1)) for k, v in cost_components_nominal[t].items()
        }

    return {
        "objective": model.ObjVal,
        "cost_per_year": cost_per_year,
        "cost_per_year_nominal": cost_per_year_nominal,
        "cost_components_nominal": cost_components_nominal,
        "cost_components_discounted": cost_components_discounted,
        "w": w_val,
        "w_on": w_on_val,
        "b": b_val,
        "b_on": b_on_val,
        "x_on": x_on_val,
        "x_off": x_off_val,
        "y": y_val,
        "x": x_val,
        "f": f_val,
        "r": r_val,
        "z": z_val,
        "P": P_val
    }
