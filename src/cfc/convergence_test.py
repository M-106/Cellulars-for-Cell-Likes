# -----------
# > Imports <
# -----------
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



# ----------
# > Helper <
# ----------
def read_values(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()["scalars"]
    print(tags)

    events = ea.Scalars("Balanced_Accuracy/Validation")

    steps = [e.step for e in events]
    values = [e.value for e in events]

    return values, steps



def analyse(values, n_last_steps=10):
    best_value = np.max(np.array(values))
    vals = np.asarray(values[-n_last_steps:])

    x = np.arange(len(vals))

    slope, intercept = np.polyfit(x, vals, 1)

    diffs = np.diff(vals)

    return {
        # "delta": vals[-1] - vals[0],
        "slope": slope,
        # "mean_diff": diffs.mean(),
        # "std_diff": diffs.std(),
        # "last": vals[-1],
        "best": best_value,
        "n_last_steps": n_last_steps
    }



# ---------
# > Main <
# ---------
def main(config):

    print("Start checking convergence.")

    root_path = "./output/"

    str_results = []

    for cur_dir in os.listdir(root_path):
        cur_full_path = os.path.join(root_path, cur_dir, "logs")

        try:
            cur_values, _ = read_values(cur_full_path)

            result = analyse(cur_values, n_last_steps=15)

            cur_str_result = f"{cur_dir}\n"
            for key, value in result.items():
                try:
                    cur_str_result += f"\n    - {key}: {value:.6f}"
                except Exception:
                    cur_str_result += f"\n    - {key}: {value}"

            str_results.append(cur_str_result)
        except Exception as e:
            print(f"Error Occured in '{cur_dir}': {e}")


    # save results
    with open("./convergence_info.txt", "w") as file_:
        file_.write("\n\n".join(str_results))

    print(f"Saved results at: './convergence_info.txt'")











