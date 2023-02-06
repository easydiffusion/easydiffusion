'''
This script checks if the given modules exist
'''

import sys
import pkgutil

modules = sys.argv[1:]
missing_modules = []
for m in modules:
    if pkgutil.find_loader(m) is None:
        print('module', m, 'not found')
        exit(1)
