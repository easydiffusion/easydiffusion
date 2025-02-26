import os
import argparse
import sys
import shutil

# The config file is in the same directory as this script
config_directory = os.path.dirname(__file__)
config_yaml = os.path.join(config_directory, "..", "config.yaml")
config_yaml = os.path.abspath(config_yaml)
config_json = os.path.join(config_directory, "config.json")

parser = argparse.ArgumentParser(description='Get values from config file')
parser.add_argument('--default', dest='default', action='store',
                     help='default value, to be used if the setting is not defined in the config file')
parser.add_argument('key', metavar='key', nargs='+',
                    help='config key to return')

args = parser.parse_args()

config = None

# migrate the old config yaml location
config_legacy_yaml = os.path.join(config_directory, "config.yaml")
if os.path.isfile(config_legacy_yaml):
    shutil.move(config_legacy_yaml, config_yaml)

if os.path.isfile(config_yaml):
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    with open(config_yaml, 'r') as configfile:
        try:
            config = yaml.load(configfile)
        except Exception as e:
            print(e, file=sys.stderr)
elif os.path.isfile(config_json):
    import json
    with open(config_json, 'r') as configfile:
        try:
            config = json.load(configfile)
        except Exception as e:
            print(e, file=sys.stderr)

if config is None:
    config = {}

for k in args.key:
    if k in config:
        config = config[k]
    else:
        if args.default != None:
            print(args.default)
            exit()

print(config)
