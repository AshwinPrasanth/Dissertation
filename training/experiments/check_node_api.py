from pyscipopt import Model


print("=" * 80)
print("MODEL METHODS")
print("=" * 80)

for name in dir(Model):

    if (
        "child" in name.lower()
        or
        "node" in name.lower()
        or
        "branch" in name.lower()
    ):

        print(name)