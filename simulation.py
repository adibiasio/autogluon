from stopping_simulator.EarlyStoppingSimulator import StoppingSimulator
import pandas as pd
import time, os


base_dir = "simulation"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

start = time.time()
with open(f"{base_dir}/time.txt", "a") as f:
    print(f"Start Time: {start}", file=f)

rangeConfigs = {
    "simple_patience": { "patience": (0, 750, 1) }, # 750 simple patience configs
    "polynomial_adaptive_patience": { "a": (0.01, 1.01, 0.05), "b": (0, 500, 5), "degree": (0.1, 1.5, 0.02) }, # (100 * 500 * 140) / 50 poly configs
}

simulator = StoppingSimulator()
simulator.load_curves("learning_curves/10/")

with open(f"{base_dir}/time.txt", "a") as f:
    print(f"Load Curves Time: {time.time()}", file=f)

simulations = simulator.run(strategies=rangeConfigs)
simulations.to_csv(f"{base_dir}/simulations.csv")

with open(f"{base_dir}/time.txt", "a") as f:
    print(f"Run Time: {time.time()}", file=f)

val_ranks = simulator.rank(eval_set="val")
val_ranks.to_csv(f"{base_dir}/val_ranks.csv")

with open(f"{base_dir}/time.txt", "a") as f:
    print(f"Val Rank Time: {time.time()}", file=f)

test_ranks = simulator.rank(eval_set="test")
test_ranks.to_csv(f"{base_dir}/test_ranks.csv")

with open(f"{base_dir}/time.txt", "a") as f:
    print(f"Test Rank Time: {time.time()}", file=f)

end = time.time()
with open(f"{base_dir}/time.txt", "a") as f:
    print(f"End Time: {end}", file=f)
    print(f"Total Time: {end-start}", file=f)
