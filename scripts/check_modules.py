"""
This script checks if the given modules exist

E.g. python check_modules.py sdkit==1.0.3 sdkit.models ldm transformers numpy antlr4 gfpgan realesrgan
"""

import sys
import pkgutil
from importlib.metadata import version

modules = sys.argv[1:]
missing_modules = []
for m in modules:
    m = m.split("==")
    module_name = m[0]
    module_version = m[1] if len(m) > 1 else None
    is_installed = pkgutil.find_loader(module_name) is not None
    if not is_installed:
        print("module", module_name, "not found")
        exit(1)
    elif module_version and version(module_name) != module_version:
        print("module version is different! expected: ", module_version, ", actual: ", version(module_name))
        exit(1)
