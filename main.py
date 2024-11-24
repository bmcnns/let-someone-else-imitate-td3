import minari
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


from data_processor import create_offline_dataset_from_minari

from sklearn.model_selection import train_test_split
from pyoperon.sklearn import SymbolicRegressor

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    offline_data = create_offline_dataset_from_minari(
        minari.load_dataset("HalfCheetah-Expert-v2"),
    ).dataset

    X, y, _, _, _ = offline_data.tensors
    y = y[:, 0]
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    values = []
    for _ in range(1):
        reg = SymbolicRegressor(
            allowed_symbols='add,sub,mul,aq,sin,constant,variable',
            offspring_generator='basic',
            optimizer_iterations=10,
            max_length=50,
            initialization_method='btc',
            n_threads=8,
            objectives=['r2', 'length'],
            epsilon=0,
            random_state=None,
            reinserter='keep-best',
            max_evaluations=int(1e6),
            symbolic_mode=False,
            tournament_size=3
        )

        reg.fit(X_train, y_train)

        values += [t['objective_values'] for t in reg.pareto_front_]

    with open('action1.pkl', 'wb') as f:
        pickle.dump((reg, values), f)

    values = np.array(values)
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.grid(True, linestyle='dotted')
    ax.set(xlabel='Obj 1 (Tree length)', ylabel='Obj 2 (-R2)')
    sns.scatterplot(ax=ax, x=values[:, 1], y=values[:, 0])