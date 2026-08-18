import sys, subprocess
from importlib.metadata import version
from packaging.specifiers import SpecifierSet

REQ = "torchruntime~=2.4.0"
NAME, VER = REQ.split("~=")
try:
    match = SpecifierSet(f"~={VER}").contains(version(NAME))
except Exception:
    match = False

if not match:
    subprocess.run([sys.executable, "-m", "pip", "install", REQ])
