import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open('action1.pkl', 'rb') as f:
    regressor, values = pickle.load(f)

print(values)

values = np.array(values)
fig, ax = plt.subplots(figsize=(18,8))
ax.grid(True, linestyle='dotted')
ax.set(xlabel='Obj 1 (Tree length)', ylabel='Obj 2 (-R2)')
sns.scatterplot(ax=ax, x=values[:,1], y=values[:,0])
plt.show()

print(regressor)

reg = regressor
res = [(s['objective_values'], s['tree'], s['minimum_description_length']) for s in reg.pareto_front_]
for obj, expr, mdl in res:
    print(obj, mdl, reg.get_model_string(expr, 16))