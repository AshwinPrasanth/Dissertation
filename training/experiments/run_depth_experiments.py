import os
import sys

# 1. This gets the 'experiments/' folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. This gets the immediate parent folder, which is 'anytime/'
anytime_dir = os.path.dirname(current_dir)

# 3. Add 'anytime/' to the Python path
if anytime_dir not in sys.path:
    sys.path.insert(0, anytime_dir)



import csv

from solver_runner import solve_instance



GRAPH = "graphs/bhoslib/frb30-15-1.clq"

DEPTHS = [

    0,

    2,

    5,
    
    7,

    10,

    -1,

]


with open(

    "results/depth_experiments.csv",

    "w",

    newline="",

) as f:

    writer = csv.writer(f)

    writer.writerow(

        [

            "depth_limit",

            "runtime",

            "nodes",

            "objective",

            "search_depth",

            "branch_calls",

        ]

    )

    for depth in DEPTHS:

        print(

            "\n======================"

        )

        print(

            "Running depth",

            depth,

        )

        result = solve_instance(

            GRAPH,

            depth,

        )

        print(result)

        writer.writerow(

            [

                result["depth_limit"],

                result["runtime"],

                result["nodes"],

                result["objective"],

                result["search_depth"],

                result["branch_calls"],

            ]

        )