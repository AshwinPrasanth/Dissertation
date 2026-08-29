from pyscipopt.scip import Event
import inspect

print("=" * 80)
print("Event methods")
print("=" * 80)

for name in dir(Event):
    if not name.startswith("_"):
        attr = getattr(Event, name)
        if callable(attr):
            try:
                print(f"{name}{inspect.signature(attr)}")
            except Exception:
                print(name)