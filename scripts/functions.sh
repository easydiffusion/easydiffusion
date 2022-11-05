#
# utility functions for all scripts
#

fail() {
    echo
    echo "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
    echo
    if [ "$1" != "" ]; then
        echo ERROR: $1
    else
        echo An error occurred.
    fi
    cat <<EOF

Error downloading Stable Diffusion UI. Sorry about that, please try to:
 1. Run this installer again.
 2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/cmdr2/stable-diffusion-ui/wiki/Troubleshooting
 3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
 4. If that doesn't solve the problem, please file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues

Thanks!


EOF
    read -p "Press any key to continue"
    exit 1

}



