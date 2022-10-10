'''
This script is run by the `installer.helpers.modules_exist_in_env()` function
'''

import sys
import pkgutil

modules = sys.argv[1:]
missing_modules = []
for m in modules:
    if pkgutil.find_loader(m) is None:
        missing_modules.append(m)

if len(missing_modules) == 0:
    print('42')
    exit()

print('Missing modules', missing_modules)