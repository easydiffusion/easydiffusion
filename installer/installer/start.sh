#!/bin/bash

# Never edit this file. If you really, really have to, beware that a script doesn't like
# being overwritten while it is running (the auto-updater will do that).
# The trick is to update this file while another script is running, and vice versa.

python $SD_BASE_DIR/installer/installer/main.py

read -p "Press enter to continue"