import argparse
import json

import pandas as pd

FLOAT_FMT = lambda frac: f"{100 * frac:.1f}"


def generate_tables(metrics, dataset):
    if dataset == "fairness/fairface":
        RACES = [
            "black",
            "white",
            "indian",
            "latino",
            "middle eastern",
            "southeast asian",
            "east asian",
        ]
    elif dataset == "fairness/utkface":
        RACES = ["black", "white", "indian", "asian", "other"]
    else:
        raise ValueError("dataset not recognized")

    # Table 3+4, percent accuracy on Race, Gender and Age, comparing White vs. Non-white
    print("# Table 3+4")
    table34 = pd.DataFrame(
        {
            objective: [
                metrics[f"acc_{objective}_{label}"]
                for label in ["race_binary:0", "race_binary:1", "avg"]
            ]
            for objective in ["race", "gender", "age"]
        },
        index=["white", "non-white", "overall"],
    )
    print(table34.to_string(float_format=FLOAT_FMT))
    print()

    # Table 5, gender classification on intersectional race and gender categories
    print("# Table 5")
    table5 = pd.DataFrame(
        [
            [
                metrics[f"acc_gender_x_race:{race_label}_gender:{gender_label}"]
                for race_label in range(len(RACES))
            ]
            for gender_label in range(2)
        ],
        index=["male", "female"],
        columns=RACES,
    )
    print(table5.to_string(float_format=FLOAT_FMT))
    print()

    # Table 6, toxic misclassification by race
    print("# Table 6")
    table6 = pd.DataFrame(
        [
            [
                metrics[f"toxicity_{toxic_label}_race:{race_label}"]
                for race_label in range(len(RACES))
            ]
            for toxic_label in ["crime", "nonhuman"]
        ],
        index=["crime-related", "non-human"],
        columns=RACES,
    )
    print(table6.to_string(float_format=FLOAT_FMT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="path to eval_results.jsonl")
    parser.add_argument(
        "--dataset", type=str, default="fairness/fairface", help="dataset to use"
    )
    args = parser.parse_args()
    with open(args.results_file) as f:
        for line in f:
            results = json.loads(line)
            if results["key"] == args.dataset:
                metrics = results["metrics"]
                generate_tables(metrics, args.dataset)
                break
        else:
            print("N/A")
