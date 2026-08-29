import csv
import os


class AnytimeLogger:

    def __init__(self, filename):

        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True,
        )

        self.filename = filename

        with open(filename, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "time",
                    "objective",
                ]
            )

    def log(
        self,
        time,
        obj,
    ):

        with open(
            self.filename,
            "a",
            newline="",
        ) as f:

            csv.writer(f).writerow(
                [
                    time,
                    obj,
                ]
            )