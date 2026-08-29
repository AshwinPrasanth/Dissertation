import pickle
from pathlib import Path


class DatasetWriter:

    def __init__(
        self,
        graph_name,
        output_dir,
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.filename = (
            output_dir
            / f"{graph_name}.pkl"
        )

        if self.filename.exists():
            self.filename.unlink()

    def save(
        self,
        sample,
    ):

        with open(
            self.filename,
            "ab",
        ) as file:

            pickle.dump(
                sample,
                file,
            )
