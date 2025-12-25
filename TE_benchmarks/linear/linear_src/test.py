import gurobipy as grb

def my_callback(model, where):
    if where == grb.GRB.Callback.MIPNODE:
        node_solution = model.cbGetNodeRel(model._vars)
        print(f"Node solution: {node_solution}")
        # Example: add a constraint to encourage deeper branching
        if node_solution[0] > 0.5:  # Assumed meaningful condition
            model.cbLazy(model._vars[0] <= 0.5)  # Add lazy constraint to expand more nodes

# Build model
model = grb.Model()

# Add extra variables and constraints to increase complexity
x = model.addVar(vtype=grb.GRB.INTEGER, name="x")
y = model.addVar(vtype=grb.GRB.INTEGER, name="y")
z = model.addVar(vtype=grb.GRB.BINARY, name="z")
w = model.addVar(vtype=grb.GRB.BINARY, name="w")

model.addConstr(2 * x + 3 * y + z <= 12, "c0")
model.addConstr(x - y + w >= 1, "c1")
model.addConstr(x + y + z <= 10, "c2")
model.addConstr(x <= 5 * z + 2 * w, "c3")
model.addConstr(w + z == 1, "c4")

model.setObjective(x + y + z + w, grb.GRB.MAXIMIZE)

# Store variables for callback access
model._vars = [x, y, z, w]

# Disable presolve and set MIPGap to force more branching
model.setParam('Presolve', 0)
model.setParam('MIPGap', 0.1)
model.setParam('NodeLimit', 10)  # Force at least 10 nodes to expand

model.setParam('Heuristics', 0)
# Optimize the model and use callback
model.optimize(my_callback)





