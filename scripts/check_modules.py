"""
This script checks and installs the required modules.

This script runs inside the legacy "stable-diffusion" folder

TODO - Maybe replace the bulk of this script with a call to `pip install -f requirements.txt`, with
a custom index URL depending on the platform.

"""

import os, sys
from importlib.metadata import version as pkg_version
import platform
import traceback
import shutil
from pathlib import Path

os_name = platform.system()

modules_to_check = {
    "torch": ("1.11.0", "1.13.1", "2.0.0"),
    "torchvision": ("0.12.0", "0.14.1", "0.15.1"),
    "sdkit": "1.0.167",
    "stable-diffusion-sdkit": "2.1.4",
    "rich": "12.6.0",
    "uvicorn": "0.19.0",
    "fastapi": "0.85.1",
    "pycloudflared": "0.2.0",
    "ruamel.yaml": "0.17.21",
    # "xformers": "0.0.16",
}
modules_to_log = ["torch", "torchvision", "sdkit", "stable-diffusion-sdkit"]


def version(module_name: str) -> str:
    try:
        return pkg_version(module_name)
    except:
        return None


def install(module_name: str, module_version: str):
    if module_name == "xformers" and (os_name == "Darwin" or is_amd_on_linux()):
        return

    index_url = None
    if module_name in ("torch", "torchvision"):
        module_version, index_url = apply_torch_install_overrides(module_version)

    if is_amd_on_linux():  # hack until AMD works properly on torch 2.0 (avoids black images on some cards)
        if module_name == "torch":
            module_version = "1.13.1+rocm5.2"
        elif module_name == "torchvision":
            module_version = "0.14.1+rocm5.2"
    elif os_name == "Darwin":
        if module_name == "torch":
            module_version = "1.13.1"
        elif module_name == "torchvision":
            module_version = "0.14.1"

    install_cmd = f"python -m pip install --upgrade {module_name}=={module_version}"
    if index_url:
        install_cmd += f" --index-url {index_url}"
    if module_name == "sdkit" and version("sdkit") is not None:
        install_cmd += " -q"

    print(">", install_cmd)
    os.system(install_cmd)


def update_modules():
    for module_name, allowed_versions in modules_to_check.items():
        if os.path.exists(f"../src/{module_name}"):
            print(f"Skipping {module_name} update, since it's in developer/editable mode")
            continue

        allowed_versions, latest_version = get_allowed_versions(module_name, allowed_versions)

        requires_install = False
        if module_name in ("torch", "torchvision"):
            if version(module_name) is None:  # allow any torch version
                requires_install = True
            elif os_name == "Darwin" and (  # force mac to downgrade from torch 2.0
                version("torch").startswith("2.") or version("torchvision").startswith("0.15.")
            ):
                requires_install = True
        elif version(module_name) not in allowed_versions:
            requires_install = True

        if requires_install:
            try:
                install(module_name, latest_version)
            except:
                traceback.print_exc()
                fail(module_name)
            else:
                if version(module_name) != latest_version:
                    print(f"WARNING! Tried to install {module_name}=={latest_version}, but the version is still {version(module_name)}!")


        if module_name in modules_to_log:
            print(f"{module_name}: {version(module_name)}")


### utilities


def get_allowed_versions(module_name: str, allowed_versions: tuple):
    allowed_versions = (allowed_versions,) if isinstance(allowed_versions, str) else allowed_versions
    latest_version = allowed_versions[-1]

    if module_name in ("torch", "torchvision"):
        allowed_versions = include_cuda_versions(allowed_versions)

    return allowed_versions, latest_version


def apply_torch_install_overrides(module_version: str):
    index_url = None
    if os_name == "Windows":
        module_version += "+cu117"
        index_url = "https://download.pytorch.org/whl/cu117"
    elif is_amd_on_linux():
        index_url = "https://download.pytorch.org/whl/rocm5.2"

    return module_version, index_url


def include_cuda_versions(module_versions: tuple) -> tuple:
    "Adds CUDA-specific versions to the list of allowed version numbers"

    allowed_versions = tuple(module_versions)
    allowed_versions += tuple(f"{v}+cu116" for v in module_versions)
    allowed_versions += tuple(f"{v}+cu117" for v in module_versions)
    allowed_versions += tuple(f"{v}+rocm5.2" for v in module_versions)
    allowed_versions += tuple(f"{v}+rocm5.4.2" for v in module_versions)

    return allowed_versions


def is_amd_on_linux():
    if os_name == "Linux":
        try:
            with open("/proc/bus/pci/devices", "r") as f:
                device_info = f.read()
                if "amdgpu" in device_info and "nvidia" not in device_info:
                    return True
        except:
            return False

    return False


def fail(module_name):
    print(
        f"""Error installing {module_name}. Sorry about that, please try to:
1. Run this installer again.
2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/easydiffusion/easydiffusion/wiki/Troubleshooting
3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
4. If that doesn't solve the problem, please file an issue at https://github.com/easydiffusion/easydiffusion/issues
Thanks!"""
    )
    exit(1)

### Launcher

def get_config():
    # The config file is in the same directory as this script
    config_directory = os.path.dirname(__file__)
    config_yaml = os.path.join(config_directory, "..", "config.yaml")
    config_json = os.path.join(config_directory, "config.json")

    config = None

    # Defaults
    listen_port = 9000
    bind_ip = "127.0.0.1"

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

    return config

def launch_uvicorn():
    config = get_config()

    print(config)

    with open("scripts/install_status.txt","a") as f:
        f.write("sd_weights_downloaded\n")
        f.write("sd_install_complete\n")

    print("\n\nEasy Diffusion installation complete, starting the server!\n\n")

    os.environ["SD_PATH"] = Path(Path.cwd(), "stable-diffusion")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = 1
    os.environ["PYTHONPATH"] = Path( os.environ["INSTALL_ENV_DIR"], "lib", "python3.8", "site-packages")
    os.environ["SD_UI_PATH"] = Path(Path.cwd(), "ui")

    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    print(f"Python:  {shutil.which('python')}")
    print(f"Version: {platform. python_version()}")

    if "net" in config:
        if "listen_port" in config["net"]:
            listen_port = config["net"]["listen_port"]
        if "listen_to_network" in config["net"] and config["net"]["listen_to_network"] == "True":
            if "bind_ip" in config["net"]:
                bind_ip = config["net"]["bind_ip"]
            else:
                bind_ip = "0.0.0.0"

    os.chdir("stable-diffusion")

    if is_amd_on_linux():
        setup_amd_environment()

    print("Launching uvicorn")
    execlp("python", "-m", "uvicorn", 
           "main:server_api",
           "--app-dir", os.environ["SD_UI_PATH"],
           "--port", listen_port,
           "--host", bind_ip,
           "--log-level", "error")

### Start
AMD_PCI_IDs = {
    "AC0C": "Theater 506A USB",
    "AC0D": "Theater 506A USB",
    "AC0E": "Theater 506A External USB",
    "AC0F": "Theater 506A External USB",
    "AC12": "Theater HD T507 (DVB-T) TV tuner/capture device",
    "AC02": "TV Wonder HD 600 PCIe",
    "AC03": "Theater 506 PCIe",
    "AC04": "Theater 506 USB",
    "AC05": "Theater 506 USB",
    "AC06": "Theater 506 External USB",
    "AC07": "Theater 506 External USB",
    "AC08": "Theater 506A World-Wide Analog Decoder + Demodulator",
    "AC09": "Theater 506A World-Wide Analog Decoder + Demodulator",
    "AC0A": "Theater 506A PCIe",
    "AC0B": "Theater 506A PCIe",
    "AC00": "Theater 506 World-Wide Analog Decoder",
    "AC01": "Theater 506 World-Wide Analog Decoder",
    "999D": "Richland [Radeon HD 8550D]",
    "99A0": "Trinity 2 [Radeon HD 7520G]",
    "99A2": "Trinity 2 [Radeon HD 7420G]",
    "99A4": "Trinity 2 [Radeon HD 7400G]",
    "9996": "Richland [Radeon HD 8470D]",
    "9997": "Richland [Radeon HD 8350G]",
    "9998": "Richland [Radeon HD 8370D]",
    "9999": "Richland [Radeon HD 8510G]",
    "999A": "Richland [Radeon HD 8410G]",
    "999B": "Richland [Radeon HD 8310G]",
    "999C": "Richland [Radeon HD 8650D]",
    "9925": "Kingston/Clayton/Jupiter/Gladius/Montego HDMI Controller",
    "9926": "Jupiter",
    "9990": "Trinity 2 [Radeon HD 7520G]",
    "9991": "Trinity 2 [Radeon HD 7540D]",
    "9992": "Trinity 2 [Radeon HD 7420G]",
    "9993": "Trinity 2 [Radeon HD 7480D]",
    "9994": "Trinity 2 [Radeon HD 7400G]",
    "9995": "Richland [Radeon HD 8450G]",
    "9917": "Trinity [Radeon HD 7620G]",
    "9918": "Trinity [Radeon HD 7600G]",
    "9919": "Trinity [Radeon HD 7500G]",
    "991E": "Bishop [Xbox One S APU]",
    "9920": "Liverpool [Playstation 4 APU]",
    "9922": "Starshp",
    "9923": "Starsha2 [Kingston/Clayton]",
    "9924": "Gladius",
    "9909": "Trinity [Radeon HD 7500G]",
    "990A": "Trinity [Radeon HD 7500G]",
    "990B": "Richland [Radeon HD 8650G]",
    "990C": "Richland [Radeon HD 8670D]",
    "990D": "Richland [Radeon HD 8550G]",
    "990E": "Richland [Radeon HD 8570D]",
    "990F": "Richland [Radeon HD 8610G]",
    "9910": "Trinity [Radeon HD 7660G]",
    "9913": "Trinity [Radeon HD 7640G]",
    "9901": "Trinity [Radeon HD 7660D]",
    "9903": "Trinity [Radeon HD 7640G]",
    "9904": "Trinity [Radeon HD 7560D]",
    "9905": "Trinity GL [FirePro A300]",
    "9906": "Trinity GL [FirePro A320]",
    "9907": "Trinity [Radeon HD 7620G]",
    "9908": "Trinity [Radeon HD 7600G]",
    "985F": "Mullins",
    "9874": "Wani [Radeon R5/R6/R7 Graphics]",
    "9890": "Amur",
    "98C0": "Nolan",
    "98E4": "Stoney [Radeon R2/R3/R4/R5 Graphics]",
    "9900": "Trinity [Radeon HD 7660G]",
    "9858": "Mullins",
    "9859": "Mullins",
    "985A": "Mullins",
    "985B": "Mullins",
    "985C": "Mullins",
    "985D": "Mullins",
    "985E": "Mullins",
    "9852": "Mullins [Radeon R2 Graphics]",
    "9853": "Mullins [Radeon R2 Graphics]",
    "9854": "Mullins [Radeon R3E Graphics]",
    "9855": "Mullins [Radeon R6 Graphics]",
    "9856": "Mullins [Radeon R1E/R2E Graphics]",
    "9857": "Mullins [Radeon APU XX-2200M with R2 Graphics]",
    "9836": "Kabini [Radeon HD 8280 / R3 Series]",
    "9837": "Kabini [Radeon HD 8280E]",
    "9838": "Kabini [Radeon HD 8240 / R3 Series]",
    "9839": "Kabini [Radeon HD 8180]",
    "983D": "Temash [Radeon HD 8250/8280G]",
    "9850": "Mullins [Radeon R3 Graphics]",
    "9851": "Mullins [Radeon R4/R5 Graphics]",
    "9808": "Wrestler [Radeon HD 7340]",
    "9809": "Wrestler [Radeon HD 7310]",
    "980A": "Wrestler [Radeon HD 7290]",
    "9830": "Kabini [Radeon HD 8400 / R3 Series]",
    "9831": "Kabini [Radeon HD 8400E]",
    "9832": "Kabini [Radeon HD 8330]",
    "9833": "Kabini [Radeon HD 8330E]",
    "9834": "Kabini [Radeon HD 8210]",
    "9835": "Kabini [Radeon HD 8310E]",
    "9714": "RS880 [Radeon HD 4290]",
    "9715": "RS880 [Radeon HD 4250]",
    "9802": "Wrestler [Radeon HD 6310]",
    "9803": "Wrestler [Radeon HD 6310]",
    "9804": "Wrestler [Radeon HD 6250]",
    "9805": "Wrestler [Radeon HD 6250]",
    "9806": "Wrestler [Radeon HD 6320]",
    "9807": "Wrestler [Radeon HD 6290]",
    "964B": "Sumo",
    "964C": "Sumo",
    "964E": "Sumo",
    "964F": "Sumo",
    "9710": "RS880 [Radeon HD 4200]",
    "9712": "RS880M [Mobility Radeon HD 4225/4250]",
    "9713": "RS880M [Mobility Radeon HD 4100]",
    "9643": "SuperSumo [Radeon HD 6380G]",
    "9644": "SuperSumo [Radeon HD 6410D]",
    "9645": "SuperSumo [Radeon HD 6410D]",
    "9647": "Sumo [Radeon HD 6520G]",
    "9648": "Sumo [Radeon HD 6480G]",
    "9649": "SuperSumo [Radeon HD 6480G]",
    "964A": "Sumo [Radeon HD 6530D]",
    "9642": "SuperSumo [Radeon HD 6370D]",
    "9615": "RS780E [Radeon HD 3200]",
    "9610": "RS780 [Radeon HD 3200]",
    "9611": "RS780C [Radeon 3100]",
    "9612": "RS780M [Mobility Radeon HD 3200]",
    "9613": "RS780MC [Mobility Radeon HD 3100]",
    "9614": "RS780D [Radeon HD 3300]",
    "9616": "RS780L [Radeon 3000]",
    "9640": "Sumo [Radeon HD 6550D]",
    "9641": "Sumo [Radeon HD 6620G]",
    "95C4": "RV620/M82 [Mobility Radeon HD 3450/3470]",
    "95C5": "RV620 LE [Radeon HD 3450]",
    "95C6": "RV620 LE [Radeon HD 3450 AGP]",
    "95C9": "RV620 LE [Radeon HD 3450 PCI]",
    "95CC": "RV620 GL [FirePro V3700]",
    "95CD": "RV620 GL [FirePro 2450]",
    "95CF": "RV620 GL [FirePro 2260]",
    "9591": "RV635/M86 [Mobility Radeon HD 3650]",
    "9593": "RV635/M86 [Mobility Radeon HD 3670]",
    "9595": "RV635/M86 GL [Mobility FireGL V5700]",
    "9596": "RV635 PRO [Radeon HD 3650 AGP]",
    "9597": "RV635 PRO [Radeon HD 3650 AGP]",
    "9598": "RV635 [Radeon HD 3650/3750/4570/4580]",
    "9599": "RV635 PRO [Radeon HD 3650 AGP]",
    "95C0": "RV620 PRO [Radeon HD 3470]",
    "95C2": "RV620/M82 [Mobility Radeon HD 3410/3430]",
    "9586": "RV630 XT [Radeon HD 2600 XT AGP]",
    "9587": "RV630 PRO [Radeon HD 2600 PRO AGP]",
    "9588": "RV630 XT [Radeon HD 2600 XT]",
    "9589": "RV630 PRO [Radeon HD 2600 PRO]",
    "958A": "RV630 [Radeon HD 2600 X2]",
    "958B": "RV630/M76 [Mobility Radeon HD 2600 XT]",
    "958C": "RV630 GL [FireGL V5600]",
    "958D": "RV630 GL [FireGL V3600]",
    "954F": "RV710 [Radeon HD 4350/4550]",
    "9552": "RV710/M92 [Mobility Radeon HD 4330/4350/4550]",
    "9553": "RV710/M92 [Mobility Radeon HD 4530/4570/5145/530v/540v/545v]",
    "9555": "RV711/M93 [Mobility Radeon HD 4350/4550/530v/540v/545v / FirePro RG220]",
    "9557": "RV711/M93 GL [FirePro RG220]",
    "955F": "RV710/M92 [Mobility Radeon HD 4330]",
    "9580": "RV630 [Radeon HD 2600 PRO]",
    "9581": "RV630/M76 [Mobility Radeon HD 2600]",
    "9583": "RV630/M76 [Mobility Radeon HD 2600 XT/2700]",
    "950F": "R680 [Radeon HD 3870 X2]",
    "9511": "RV670 GL [FireGL V7700]",
    "9513": "RV670 [Radeon HD 3850 X2]",
    "9515": "RV670 PRO [Radeon HD 3850 AGP]",
    "9519": "RV670 GL [FireStream 9170]",
    "9540": "RV710 [Radeon HD 4550]",
    "9500": "RV670 [Radeon HD 3850 X2]",
    "9501": "RV670 [Radeon HD 3870]",
    "9504": "RV670/M88 [Mobility Radeon HD 3850]",
    "9505": "RV670 [Radeon HD 3690/3850]",
    "9506": "RV670/M88 [Mobility Radeon HD 3850 X2]",
    "9507": "RV670 [Radeon HD 3830]",
    "9508": "RV670/M88-XT [Mobility Radeon HD 3870]",
    "9509": "RV670/M88 [Mobility Radeon HD 3870 X2]",
    "94C1": "RV610 [Radeon HD 2400 PRO/XT]",
    "94C3": "RV610 [Radeon HD 2400 PRO]",
    "94C4": "RV610 LE [Radeon HD 2400 PRO AGP]",
    "94C5": "RV610 [Radeon HD 2400 LE]",
    "94C7": "RV610 [Radeon HD 2350]",
    "94C8": "RV610/M74 [Mobility Radeon HD 2400 XT]",
    "94C9": "RV610/M72-S [Mobility Radeon HD 2400]",
    "94CB": "RV610 [Radeon E2400]",
    "94CC": "RV610 LE [Radeon HD 2400 PRO PCI]",
    "9498": "RV730 PRO [Radeon HD 4650]",
    "949C": "RV730 GL [FirePro V7750]",
    "949E": "RV730 GL [FirePro V5700]",
    "949F": "RV730 GL [FirePro V3750]",
    "94A0": "RV740/M97 [Mobility Radeon HD 4830]",
    "94A1": "RV740/M97-XT [Mobility Radeon HD 4860]",
    "94A3": "RV740/M97 GL [FirePro M7740]",
    "94B3": "RV740 PRO [Radeon HD 4770]",
    "94B4": "RV740 PRO [Radeon HD 4750]",
    "9462": "RV790 [Radeon HD 4860]",
    "946A": "RV770 GL [FirePro M7750]",
    "9480": "RV730/M96 [Mobility Radeon HD 4650/5165]",
    "9488": "RV730/M96-XT [Mobility Radeon HD 4670]",
    "9489": "RV730/M96 GL [Mobility FireGL V5725]",
    "9490": "RV730 XT [Radeon HD 4670]",
    "9491": "RV730/M96-CSP [Radeon E4690]",
    "9495": "RV730 [Radeon HD 4600 AGP Series]",
    "944A": "RV770/M98L [Mobility Radeon HD 4850]",
    "944B": "RV770/M98 [Mobility Radeon HD 4850 X2]",
    "944C": "RV770 LE [Radeon HD 4830]",
    "944E": "RV770 CE [Radeon HD 4710]",
    "9450": "RV770 GL [FireStream 9270]",
    "9452": "RV770 GL [FireStream 9250]",
    "9456": "RV770 GL [FirePro V8700]",
    "945A": "RV770/M98-XT [Mobility Radeon HD 4870]",
    "9460": "RV790 [Radeon HD 4890]",
    "940A": "R600 GL [FireGL V8650]",
    "940B": "R600 GL [FireGL V8600]",
    "940F": "R600 GL [FireGL V7600]",
    "9440": "RV770 [Radeon HD 4870]",
    "9441": "R700 [Radeon HD 4870 X2]",
    "9442": "RV770 [Radeon HD 4850]",
    "9443": "R700 [Radeon HD 4850 X2]",
    "9444": "RV770 GL [FirePro V8750]",
    "9446": "RV770 GL [FirePro V7760]",
    "793F": "RS690M [Radeon Xpress 1200/1250/1270] (Secondary)",
    "7941": "RS600 [Radeon Xpress 1250]",
    "7942": "RS600M [Radeon Xpress 1250]",
    "796E": "RS740 [Radeon 2100]",
    "9400": "R600 [Radeon HD 2900 PRO/XT]",
    "9401": "R600 [Radeon HD 2900 XT]",
    "9403": "R600 [Radeon HD 2900 PRO]",
    "9405": "R600 [Radeon HD 2900 GT]",
    "791E": "RS690 [Radeon X1200]",
    "791F": "RS690M [Radeon Xpress 1200/1250/1270]",
    "7835": "RS350M [Mobility Radeon 9000 IGP]",
    "7448": "Navi31 [Radeon Pro W7900]",
    "744C": "Navi 31 [Radeon RX 7900 XT/7900 XTX]",
    "745E": "Navi31 [Radeon Pro W7800]",
    "7480": "Navi 33 [Radeon RX 7700S/7600S/7600M XT]",
    "7483": "Navi 33 [Radeon RX 7600M/7600M XT]",
    "7489": "Navi 33",
    "7834": "RS350 [Radeon 9100 PRO/XT IGP]",
    "7446": "Navi 31 USB",
    "743F": "Navi 24 [Radeon RX 6400/6500 XT/6500M]",
    "7421": "Navi 24 [Radeon PRO W6500M]",
    "7422": "Navi 24 [Radeon PRO W6400]",
    "7423": "Navi 24 [Radeon PRO W6300/W6300M]",
    "7424": "Navi 24 [Radeon RX 6300]",
    "73F0": "Navi 33 [Radeon RX 7600M XT]",
    "73EF": "Navi 23 [Radeon RX 6650 XT / 6700S / 6800S]",
    "73E1": "Navi 23 WKS-XM [Radeon PRO W6600M]",
    "73FF": "Navi 23 [Radeon RX 6600/6600 XT/6600M]",
    "73E3": "Navi 23 WKS-XL [Radeon PRO W6600]",
    "73E4": "Navi 23 USB",
    "7408": "Aldebaran/MI200 [Instinct MI250X]",
    "740C": "Aldebaran/MI200 [Instinct MI250X/MI250]",
    "740F": "Aldebaran/MI200 [Instinct MI210]",
    "73CE": "Navi22-XL SRIOV MxGPU",
    "73AF": "Navi 21 [Radeon RX 6900 XT]",
    "73BF": "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]",
    "73C4": "Navi 22 USB",
    "73C3": "Navi 22",
    "73E0": "Navi 23",
    "73DF": "Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]",
    "73A5": "Navi 21 [Radeon RX 6950 XT]",
    "73AE": "Navi 21 [Radeon Pro V620 MxGPU]",
    "73A2": "Navi 21 Pro-XTA [Radeon Pro W6900X]",
    "73A3": "Navi 21 GL-XL [Radeon PRO W6800]",
    "73A4": "Navi 21 USB",
    "73AB": "Navi 21 Pro-XLA [Radeon Pro W6800X/Radeon Pro W6800X Duo]",
    "734F": "Navi 14 [Radeon Pro W5300M]",
    "7360": "Navi 12 [Radeon Pro 5600M/V520/BC-160]",
    "73A1": "Navi 21 [Radeon Pro V620]",
    "7362": "Navi 12 [Radeon Pro V520/V540]",
    "7388": "Arcturus GL-XL",
    "738C": "Arcturus GL-XL [Instinct MI100]",
    "738E": "Arcturus GL-XL [Instinct MI100]",
    "7319": "Navi 10 [Radeon Pro 5700 XT]",
    "731B": "Navi 10 [Radeon Pro 5700]",
    "731F": "Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT]",
    "7340": "Navi 14 [Radeon RX 5500/5500M / Pro 5500M]",
    "7341": "Navi 14 [Radeon Pro W5500]",
    "7347": "Navi 14 [Radeon Pro W5500M]",
    "731E": "TDC-150",
    "72A0": "RV570 [Radeon X1950 PRO] (Secondary)",
    "72A8": "RV570 [Radeon X1950 GT] (Secondary)",
    "72B1": "RV560 [Radeon X1650 XT] (Secondary)",
    "72B3": "RV560 [Radeon X1650 GT] (Secondary)",
    "7300": "Fiji [Radeon R9 FURY / NANO Series]",
    "7310": "Navi 10 [Radeon Pro W5700X]",
    "7312": "Navi 10 [Radeon Pro W5700]",
    "7314": "Navi 10 USB",
    "724E": "R580 GL [FireGL V7350]",
    "7269": "R580 [Radeon X1900 XT] (Secondary)",
    "726B": "R580 [Radeon X1900 GT] (Secondary)",
    "726E": "R580 [AMD Stream Processor] (Secondary)",
    "7280": "RV570 [Radeon X1950 PRO]",
    "7288": "RV570 [Radeon X1950 GT]",
    "7291": "RV560 [Radeon X1650 XT]",
    "7293": "RV560 [Radeon X1650 GT]",
    "71F2": "RV530 GL [FireGL V3400] (Secondary)",
    "7210": "RV550/M71 [Mobility Radeon HD 2300]",
    "7211": "RV550/M71 [Mobility Radeon X2300 HD]",
    "7240": "R580+ [Radeon X1950 XTX]",
    "7244": "R580+ [Radeon X1950 XT]",
    "7248": "R580 [Radeon X1950]",
    "7249": "R580 [Radeon X1900 XT]",
    "724B": "R580 [Radeon X1900 GT]",
    "71D6": "RV530/M66-XT [Mobility Radeon X1700]",
    "71DE": "RV530/M66 [Mobility Radeon X1700/X2500]",
    "71E0": "RV530 [Radeon X1600] (Secondary)",
    "71E1": "RV535 [Radeon X1650 PRO] (Secondary)",
    "71E2": "RV530 [Radeon X1600] (Secondary)",
    "71E6": "RV530 [Radeon X1650] (Secondary)",
    "71E7": "RV535 [Radeon X1650 PRO] (Secondary)",
    "71C4": "RV530/M56 GL [Mobility FireGL V5200]",
    "71C5": "RV530/M56-P [Mobility Radeon X1600]",
    "71C6": "RV530LE [Radeon X1600/X1650 PRO]",
    "71C7": "RV535 [Radeon X1650 PRO]",
    "71CE": "RV530 [Radeon X1300 XT/X1600 PRO]",
    "71D2": "RV530 GL [FireGL V3400]",
    "71D4": "RV530/M66 GL [Mobility FireGL V5250]",
    "71D5": "RV530/M66-P [Mobility Radeon X1700]",
    "71A7": "RV516 [Radeon X1300/X1550 Series] (Secondary)",
    "71BB": "RV516 GL [FireMV 2250] (Secondary)",
    "71C0": "RV530 [Radeon X1600 XT/X1650 GTO]",
    "71C1": "RV535 [Radeon X1650 PRO]",
    "71C2": "RV530 [Radeon X1600 PRO]",
    "71C3": "RV530 [Radeon X1600 PRO]",
    "718D": "RV516/M64-CSP128 [Mobility Radeon X1450]",
    "7193": "RV516 [Radeon X1550 Series]",
    "7196": "RV516/M62-S [Mobility Radeon X1350]",
    "719B": "RV516 GL [FireMV 2250]",
    "719F": "RV516 [Radeon X1550 Series]",
    "71A0": "RV516 [Radeon X1300/X1550 Series] (Secondary)",
    "71A1": "RV516 [Radeon X1600/X1650 Series] (Secondary)",
    "71A3": "RV516 [Radeon X1300/X1550 Series] (Secondary)",
    "7186": "RV516/M64 [Mobility Radeon X1450]",
    "7187": "RV516 [Radeon X1300/X1550 Series]",
    "7188": "RV516/M64-S [Mobility Radeon X2300]",
    "718A": "RV516/M64 [Mobility Radeon X2300]",
    "718B": "RV516/M62 [Mobility Radeon X1350]",
    "718C": "RV516/M62-CSP64 [Mobility Radeon X1350]",
    "7162": "RV515 PRO [Radeon X1300/X1550 Series] (Secondary)",
    "7163": "RV505 [Radeon X1550 Series] (Secondary)",
    "7166": "RV515 [Radeon X1300/X1550 Series] (Secondary)",
    "7167": "RV515 [Radeon X1550 64-bit] (Secondary)",
    "7172": "RV515 GL [FireGL V3300] (Secondary)",
    "7173": "RV515 GL [FireGL V3350] (Secondary)",
    "7181": "RV516 [Radeon X1600/X1650 Series]",
    "7183": "RV516 [Radeon X1300/X1550 Series]",
    "7145": "RV515/M54 [Mobility Radeon X1400]",
    "7146": "RV515 [Radeon X1300/X1550]",
    "7147": "RV505 [Radeon X1550 64-bit]",
    "7149": "RV515/M52 [Mobility Radeon X1300]",
    "714A": "RV515/M52 [Mobility Radeon X1300]",
    "7152": "RV515 GL [FireGL V3300]",
    "7153": "RV515 GL [FireGL V3350]",
    "715F": "RV505 CE [Radeon X1550 64-bit]",
    "7120": "R520 [Radeon X1800] (Secondary)",
    "7124": "R520 GL [FireGL V7200] (Secondary)",
    "7129": "R520 [Radeon X1800] (Secondary)",
    "712E": "R520 GL [FireGL V7300] (Secondary)",
    "712F": "R520 GL [FireGL V7350] (Secondary)",
    "7140": "RV515 [Radeon X1300/X1550/X1600 Series]",
    "7142": "RV515 PRO [Radeon X1300/X1550 Series]",
    "7143": "RV505 [Radeon X1300/X1550 Series]",
    "7102": "R520/M58 [Mobility Radeon X1800]",
    "7104": "R520 GL [FireGL V7200 / Barco MXTR-5100]",
    "7109": "R520 [Radeon X1800 XL]",
    "710A": "R520 [Radeon X1800 GTO]",
    "710B": "R520 [Radeon X1800 GTO]",
    "710E": "R520 GL [FireGL V7300]",
    "710F": "R520 GL [FireGL V7350]",
    "69A1": "Vega 12",
    "69A2": "Vega 12",
    "69A3": "Vega 12",
    "69AF": "Vega 12 [Radeon Pro Vega 20]",
    "6FDF": "Polaris 20 XL [Radeon RX 580 2048SP]",
    "7100": "R520 [Radeon X1800 XT]",
    "7101": "R520/M58 [Mobility Radeon X1800 XT]",
    "6980": "Polaris12",
    "6981": "Lexa XT [Radeon PRO WX 3200]",
    "6985": "Lexa XT [Radeon PRO WX 3100]",
    "6986": "Polaris12",
    "6987": "Lexa [Radeon 540X/550X/630 / RX 640 / E9171 MCM]",
    "6995": "Lexa XT [Radeon PRO WX 2100]",
    "699F": "Lexa PRO [Radeon 540/540X/550/550X / RX 540X/550/550X]",
    "69A0": "Vega 12",
    "698F": "Lexa XT [Radeon PRO WX 3100 / Barco MXRT 4700]",
    "692F": "Tonga XTV GL [FirePro S7150V]",
    "6938": "Tonga XT / Amethyst XT [Radeon R9 380X / R9 M295X]",
    "6939": "Tonga PRO [Radeon R9 285/380]",
    "694C": "Polaris 22 XT [Radeon RX Vega M GH]",
    "694E": "Polaris 22 XL [Radeon RX Vega M GL]",
    "694F": "Polaris 22 MGL XL [Radeon Pro WX Vega M GL]",
    "6930": "Tonga PRO [Radeon R9 380 4GB]",
    "693B": "Tonga PRO GL [FirePro W7100 / Barco MXRT-7600]",
    "68FE": "Cedar LE",
    "6900": "Topaz XT [Radeon R7 M260/M265 / M340/M360 / M440/M445 / 530/535 / 620/625 Mobile]",
    "6901": "Topaz PRO [Radeon R5 M255]",
    "6907": "Meso XT [Radeon R5 M315]",
    "6920": "Amethyst [Radeon R9 M395/ M395X Mac Edition]",
    "6921": "Amethyst XT [Radeon R9 M295X / M390X]",
    "6929": "Tonga XT GL [FirePro S7150]",
    "692B": "Tonga PRO GL [FirePro W7100]",
    "68FA": "Cedar [Radeon HD 7350/8350 / R5 220]",
    "68E4": "Robson CE [Radeon HD 6370M/7370M]",
    "68E5": "Robson LE [Radeon HD 6330M]",
    "68E8": "Cedar",
    "68E9": "Cedar [ATI FirePro (FireGL) Graphics Adapter]",
    "68F1": "Cedar GL [FirePro 2460]",
    "68F2": "Cedar GL [FirePro 2270]",
    "68F8": "Cedar [Radeon HD 7300 Series]",
    "68F9": "Cedar [Radeon HD 5000/6000/7350/8350 Series]",
    "68C8": "Redwood XT GL [FirePro V4800]",
    "68C9": "Redwood PRO GL [FirePro V3800]",
    "68D8": "Redwood XT [Radeon HD 5670/5690/5730]",
    "68D9": "Redwood PRO [Radeon HD 5550/5570/5630/6510/6610/7570]",
    "68DA": "Redwood LE [Radeon HD 5550/5570/5630/6390/6490/7570]",
    "68DE": "Redwood",
    "68E0": "Park [Mobility Radeon HD 5430/5450/5470]",
    "68E1": "Park [Mobility Radeon HD 5430]",
    "68A9": "Juniper XT [FirePro V5800]",
    "68B8": "Juniper XT [Radeon HD 5770]",
    "68B9": "Juniper LE [Radeon HD 5670 640SP Edition]",
    "68BA": "Juniper XT [Radeon HD 6770]",
    "68BE": "Juniper PRO [Radeon HD 5750]",
    "68BF": "Juniper PRO [Radeon HD 6750]",
    "68C0": "Madison [Mobility Radeon HD 5730 / 6570M]",
    "68C1": "Madison [Mobility Radeon HD 5650/5750 / 6530M/6550M]",
    "68C7": "Pinewood [Mobility Radeon HD 5570/6550A]",
    "6898": "Cypress XT [Radeon HD 5870]",
    "6899": "Cypress PRO [Radeon HD 5850]",
    "689B": "Cypress PRO [Radeon HD 6800 Series]",
    "689C": "Hemlock [Radeon HD 5970]",
    "689D": "Hemlock [Radeon HD 5970]",
    "689E": "Cypress LE [Radeon HD 5830]",
    "68A0": "Broadway XT [Mobility Radeon HD 5870]",
    "68A1": "Broadway PRO [Mobility Radeon HD 5850]",
    "68A8": "Granville [Radeon HD 6850M/6870M]",
    "686D": "Vega 10 GLXTA",
    "686E": "Vega 10 GLXLA",
    "687F": "Vega 10 XL/XT [Radeon RX Vega 56/64]",
    "6880": "Lexington [Radeon HD 6550M]",
    "6888": "Cypress XT [FirePro V8800]",
    "6889": "Cypress PRO [FirePro V7800]",
    "688A": "Cypress XT [FirePro V9800]",
    "688C": "Cypress XT GL [FireStream 9370]",
    "688D": "Cypress PRO GL [FireStream 9350]",
    "6869": "Vega 10 XGA [Radeon Pro Vega 48]",
    "686A": "Vega 10 LEA",
    "686B": "Vega 10 XTXA [Radeon Pro Vega 64X]",
    "6862": "Vega 10 XT [Radeon PRO SSG]",
    "6863": "Vega 10 XTX [Radeon Vega Frontier Edition]",
    "6864": "Vega 10 [Radeon Pro V340/Instinct MI25x2]",
    "6867": "Vega 10 XL [Radeon Pro Vega 56]",
    "6868": "Vega 10 [Radeon PRO WX 8100/8200]",
    "686C": "Vega 10 [Instinct MI25 MxGPU/MI25x2 MxGPU/V340 MxGPU/V340L MxGPU]",
    "683F": "Cape Verde PRO [Radeon HD 7750/8740 / R7 250E]",
    "6840": "Thames [Radeon HD 7500M/7600M Series]",
    "6841": "Thames [Radeon HD 7550M/7570M/7650M]",
    "6842": "Thames LE [Radeon HD 7000M Series]",
    "6843": "Thames [Radeon HD 7670M]",
    "6860": "Vega 10 [Instinct MI25/MI25x2/V340/V320]",
    "6861": "Vega 10 XT [Radeon PRO WX 9100]",
    "6828": "Cape Verde PRO [FirePro W600]",
    "6829": "Cape Verde",
    "682A": "Venus PRO",
    "682B": "Cape Verde PRO / Venus LE / Tropo PRO-L [Radeon HD 8830M / R7 250 / R7 M465X]",
    "682C": "Cape Verde GL [FirePro W4100]",
    "682D": "Chelsea XT GL [FirePro M4000]",
    "682F": "Chelsea LP [Radeon HD 7730M]",
    "6835": "Cape Verde PRX [Radeon R9 255 OEM]",
    "6837": "Cape Verde LE [Radeon HD 7730/8730]",
    "683D": "Cape Verde XT [Radeon HD 7770/8760 / R7 250X]",
    "6820": "Venus XTX [Radeon HD 8890M / R9 M275X/M375X]",
    "6821": "Venus XT [Radeon HD 8870M / R9 M270X/M370X]",
    "6822": "Venus PRO [Radeon E8860]",
    "6823": "Venus PRO [Radeon HD 8850M / R9 M265X]",
    "6825": "Heathrow XT [Radeon HD 7870M]",
    "6826": "Chelsea LP [Radeon HD 7700M Series]",
    "6827": "Heathrow PRO [Radeon HD 7850M/8850M]",
    "6808": "Pitcairn XT GL [FirePro W7000]",
    "6809": "Pitcairn LE GL [FirePro W5000]",
    "6810": "Curacao XT / Trinidad XT [Radeon R7 370 / R9 270X/370X]",
    "6811": "Curacao PRO [Radeon R7 370 / R9 270/370 OEM]",
    "6816": "Pitcairn",
    "6817": "Pitcairn",
    "6818": "Pitcairn XT [Radeon HD 7870 GHz Edition]",
    "6819": "Pitcairn PRO [Radeon HD 7850 / R7 265 / R9 270 1024SP]",
    "67EB": "Baffin [Radeon Pro V5300X]",
    "67EF": "Baffin [Radeon RX 460/560D / Pro 450/455/460/555/555X/560/560X]",
    "67FF": "Baffin [Radeon RX 550 640SP / RX 560/560X]",
    "6800": "Wimbledon XT [Radeon HD 7970M]",
    "6801": "Neptune XT [Radeon HD 8970M]",
    "6802": "Wimbledon",
    "6806": "Neptune",
    "67DF": "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]",
    "67E0": "Baffin [Radeon Pro WX 4170]",
    "67E1": "Baffin [Polaris11]",
    "67E3": "Baffin [Radeon Pro WX 4100]",
    "67E8": "Baffin [Radeon Pro WX 4130/4150]",
    "67E9": "Baffin [Polaris11]",
    "67D7": "Ellesmere [Radeon Pro WX 5100 / Barco MXRT-6700]",
    "67C7": "Ellesmere [Radeon Pro WX 5100]",
    "67CA": "Ellesmere [Polaris10]",
    "67CC": "Ellesmere [Polaris10]",
    "67CF": "Ellesmere [Polaris10]",
    "67D0": "Ellesmere [Radeon Pro V7300X / V7350x2]",
    "67D4": "Ellesmere [Radeon Pro WX 7100 / Barco MXRT-8700]",
    "67B9": "Vesuvius [Radeon R9 295X2]",
    "67BE": "Hawaii LE",
    "67C0": "Ellesmere [Radeon Pro WX 7100 Mobile]",
    "67C2": "Ellesmere [Radeon Pro V7300X / V7350x2]",
    "67C4": "Ellesmere [Radeon Pro WX 7100]",
    "67B8": "Hawaii XT [Radeon R9 290X Engineering Sample]",
    "67A2": "Hawaii GL",
    "67A8": "Hawaii",
    "67A9": "Hawaii",
    "67AA": "Hawaii",
    "67B0": "Hawaii XT / Grenada XT [Radeon R9 290X/390X]",
    "67B1": "Hawaii PRO [Radeon R9 290/390]",
    "679A": "Tahiti PRO [Radeon HD 7950/8950 OEM / R9 280]",
    "679B": "Malta [Radeon HD 7990/8990 OEM]",
    "679E": "Tahiti LE [Radeon HD 7870 XT]",
    "679F": "Tahiti",
    "67A0": "Hawaii XT GL [FirePro W9100]",
    "67A1": "Hawaii PRO GL [FirePro W8100]",
    "6780": "Tahiti XT GL [FirePro W9000]",
    "6784": "Tahiti [FirePro Series Graphics Adapter]",
    "6788": "Tahiti [FirePro Series Graphics Adapter]",
    "678A": "Tahiti PRO GL [FirePro Series]",
    "6798": "Tahiti XT [Radeon HD 7970/8970 OEM / R9 280X]",
    "6770": "Caicos [Radeon HD 6450A/7450A]",
    "6771": "Caicos XTX [Radeon HD 8490 / R5 235X OEM]",
    "6772": "Caicos [Radeon HD 7450A]",
    "6778": "Caicos XT [Radeon HD 7470/8470 / R5 235/310 OEM]",
    "6779": "Caicos [Radeon HD 6450/7450/8450 / R5 230 OEM]",
    "677B": "Caicos PRO [Radeon HD 7450]",
    "6761": "Seymour LP [Radeon HD 6430M]",
    "6763": "Seymour [Radeon E6460]",
    "6764": "Seymour [Radeon HD 6400M Series]",
    "6765": "Seymour [Radeon HD 6400M Series]",
    "6766": "Caicos",
    "6767": "Caicos",
    "6768": "Caicos",
    "6759": "Turks PRO [Radeon HD 6570/7570/8550 / R5 230]",
    "675B": "Turks [Radeon HD 7600 Series]",
    "675D": "Turks PRO [Radeon HD 7570]",
    "675F": "Turks LE [Radeon HD 5570/6510/7510/8510]",
    "6760": "Seymour [Radeon HD 6400M/7400M Series]",
    "6743": "Whistler [Radeon E6760]",
    "6749": "Turks GL [FirePro V4900]",
    "674A": "Turks GL [FirePro V3900]",
    "6750": "Onega [Radeon HD 6650A/7650A]",
    "6751": "Turks [Radeon HD 7650A/7670A]",
    "6758": "Turks XT [Radeon HD 6670/7670]",
    "671F": "Cayman CE [Radeon HD 6930]",
    "6720": "Blackcomb [Radeon HD 6970M/6990M]",
    "6738": "Barts XT [Radeon HD 6870]",
    "6739": "Barts PRO [Radeon HD 6850]",
    "673E": "Barts LE [Radeon HD 6790]",
    "6740": "Whistler [Radeon HD 6730M/6770M/7690M XT]",
    "6741": "Whistler [Radeon HD 6630M/6650M/6750M/7670M/7690M]",
    "6742": "Whistler LE [Radeon HD 6610M/7610M]",
    "6707": "Cayman LE GL [FirePro V5900]",
    "6718": "Cayman XT [Radeon HD 6970]",
    "6719": "Cayman PRO [Radeon HD 6950]",
    "671C": "Antilles [Radeon HD 6990]",
    "671D": "Antilles [Radeon HD 6990]",
    "66A2": "Vega 20",
    "66A3": "Vega 20 [Radeon Pro Vega II/Radeon Pro Vega II Duo]",
    "66A7": "Vega 20 [Radeon Pro Vega 20]",
    "66AF": "Vega 20 [Radeon VII]",
    "6704": "Cayman PRO GL [FirePro V7900]",
    "6665": "Jet PRO [Radeon R5 M230 / R7 M260DX / Radeon 520/610 Mobile]",
    "6667": "Jet ULT [Radeon R5 M230]",
    "666F": "Sun LE [Radeon HD 8550M / R5 M230]",
    "66A0": "Vega 20 [Radeon Pro/Radeon Instinct]",
    "66A1": "Vega 20 [Radeon Pro VII/Radeon Instinct MI50 32GB]",
    "6658": "Bonaire XTX [Radeon R7 260X/360]",
    "665C": "Bonaire XT [Radeon HD 7790/8770 / R7 360 / R9 260/360 OEM]",
    "665D": "Bonaire [Radeon R7 200 Series]",
    "665F": "Tobago PRO [Radeon R7 360 / R9 360 OEM]",
    "6660": "Sun XT [Radeon HD 8670A/8670M/8690M / R5 M330 / M430 / Radeon 520 Mobile]",
    "6663": "Sun PRO [Radeon HD 8570A/8570M]",
    "6664": "Jet XT [Radeon R5 M240]",
    "6631": "Oland",
    "6640": "Saturn XT [FirePro M6100]",
    "6641": "Saturn PRO [Radeon HD 8930M]",
    "6646": "Bonaire XT [Radeon R9 M280X / FirePro W6150M]",
    "6647": "Saturn PRO/XT [Radeon R9 M270X/M280X]",
    "6649": "Bonaire [FirePro W5100]",
    "6650": "Bonaire",
    "6651": "Bonaire",
    "664D": "Bonaire [FirePro W5100 / Barco MXRT-5600]",
    "6606": "Mars XTX [Radeon HD 8790M]",
    "6607": "Mars LE [Radeon HD 8530M / R5 M240]",
    "6608": "Oland GL [FirePro W2100]",
    "6610": "Oland XT [Radeon HD 8670 / R5 340X OEM / R7 250/350/350X OEM]",
    "6611": "Oland [Radeon HD 8570 / R5 430 OEM / R7 240/340 / Radeon 520 OEM]",
    "6613": "Oland PRO [Radeon R7 240/340 / Radeon 520]",
    "6617": "Oland LE [Radeon R7 240]",
    "6609": "Oland GL [FirePro W2100 / Barco MXRT 2600]",
    "5E4F": "RV410 [Radeon X700]",
    "5E6B": "RV410 [Radeon X700 PRO] (Secondary)",
    "5E6D": "RV410 [Radeon X700] (Secondary)",
    "5F57": "R423 [Radeon X800 XT]",
    "6600": "Mars [Radeon HD 8670A/8670M/8750M / R7 M370]",
    "6601": "Mars [Radeon HD 8730M]",
    "6604": "Opal XT [Radeon R7 M265/M365X/M465]",
    "6605": "Opal PRO [Radeon R7 M260X]",
    "5D72": "R480 [Radeon X850 XT] (Secondary)",
    "5D77": "R423 [Radeon X800 XT] (Secondary)",
    "5E48": "RV410 GL [FireGL V5000]",
    "5E49": "RV410 [Radeon X700 Series]",
    "5E4A": "RV410 [Radeon X700 XT]",
    "5E4B": "RV410 [Radeon X700 PRO]",
    "5E4C": "RV410 [Radeon X700 SE]",
    "5E4D": "RV410 [Radeon X700]",
    "5D4D": "R480 [Radeon X850 XT Platinum Edition]",
    "5D4E": "R480 [Radeon X850 SE]",
    "5D4F": "R480 [Radeon X800 GTO]",
    "5D50": "R480 GL [FireGL V7200]",
    "5D52": "R480 [Radeon X850 XT]",
    "5D57": "R423 [Radeon X800 XT]",
    "5D6D": "R480 [Radeon X850 XT Platinum Edition] (Secondary)",
    "5D6F": "R480 [Radeon X800 GTO] (Secondary)",
    "5B73": "RV370 [Radeon X300/X550/X1050 Series] (Secondary)",
    "5B74": "RV370 GL [FireGL V3100] (Secondary)",
    "5B75": "RV370 GL [FireMV 2200] (Secondary)",
    "5C61": "RV280/M9+ [Mobility Radeon 9200 AGP]",
    "5C63": "RV280/M9+ [Mobility Radeon 9200 AGP]",
    "5D44": "RV280 [Radeon 9200 SE] (Secondary)",
    "5D45": "RV280 GL [FireMV 2200 PCI] (Secondary)",
    "5D48": "R423/M28 [Mobility Radeon X800 XT]",
    "5D49": "R423/M28 GL [Mobility FireGL V5100]",
    "5D4A": "R423/M28 [Mobility Radeon X800]",
    "5A62": "RC410M [Mobility Radeon Xpress 200M]",
    "5B60": "RV370 [Radeon X300]",
    "5B62": "RV370 [Radeon X600/X600 SE]",
    "5B63": "RV370 [Radeon X300/X550/X1050 Series]",
    "5B64": "RV370 GL [FireGL V3100]",
    "5B65": "RV370 GL [FireMV 2200]",
    "5B66": "RV370X",
    "5B70": "RV370 [Radeon X300 SE]",
    "5B72": "RV380 [Radeon X300/X550/X1050 Series] (Secondary)",
    "5A36": "RC4xx/RS4xx PCI Express Port 1",
    "5A37": "RC4xx/RS4xx PCI Express Port 2",
    "5A38": "RC4xx/RS4xx PCI Express Port 3",
    "5A39": "RC4xx/RS4xx PCI Express Port 4",
    "5A41": "RS400 [Radeon Xpress 200]",
    "5A42": "RS400M [Radeon Xpress 200M]",
    "5A61": "RC410 [Radeon Xpress 200/1100]",
    "5A23": "RD890S/RD990 I/O Memory Management Unit (IOMMU)",
    "5A34": "RS4xx PCI Express Port [ext gfx]",
    "5965": "RV280 GL [FireMV 2200 PCI]",
    "5974": "RS482/RS485 [Radeon Xpress 1100/1150]",
    "5975": "RS482M [Mobility Radeon Xpress 200]",
    "5960": "RV280 [Radeon 9200 PRO / 9250]",
    "5961": "RV280 [Radeon 9200]",
    "5962": "RV280 [Radeon 9200]",
    "5964": "RV280 [Radeon 9200 SE]",
    "5940": "RV280 [Radeon 9200 PRO] (Secondary)",
    "5941": "RV280 [Radeon 9200] (Secondary)",
    "5944": "RV280 [Radeon 9200 SE PCI]",
    "5954": "RS480 [Radeon Xpress 200 Series]",
    "5955": "RS480M [Mobility Radeon Xpress 200]",
    "5834": "RS300 [Radeon 9100 IGP]",
    "5835": "RS300M [Mobility Radeon 9100 IGP]",
    "5854": "RS480 [Radeon Xpress 200 Series] (Secondary)",
    "5874": "RS480 [Radeon Xpress 1150] (Secondary)",
    "564B": "RV410/M26 GL [Mobility FireGL V5000]",
    "564F": "RV410/M26 [Mobility Radeon X700 XL]",
    "5652": "RV410/M26 [Mobility Radeon X700]",
    "5653": "RV410/M26 [Mobility Radeon X700]",
    "5654": "Mach64 VT [Video Xpression]",
    "5655": "264VT3 [Mach64 VT3]",
    "5656": "Mach64 VT4 [Video Xpression+]",
    "5657": "RV410 [Radeon X550 XTX / X700]",
    "554E": "R430 [All-In-Wonder X800 GT]",
    "554F": "R430 [Radeon X800]",
    "5550": "R423 GL [FireGL V7100]",
    "5551": "R423 GL [FireGL V5100]",
    "5569": "R423 [Radeon X800 PRO] (Secondary)",
    "556B": "R423 [Radeon X800 GT] (Secondary)",
    "556D": "R480 [Radeon X800 GTO2/XL] (Secondary)",
    "556F": "R430 [Radeon X800] (Secondary)",
    "5571": "R423 GL [FireGL V5100] (Secondary)",
    "5462": "RV380/M24C [Mobility Radeon X600 SE]",
    "5464": "RV370/M22 GL [Mobility FireGL V3100]",
    "5549": "R423 [Radeon X800 GTO]",
    "554A": "R423 [Radeon X800 XT Platinum Edition]",
    "554B": "R423 [Radeon X800 GT/SE]",
    "554D": "R480 [Radeon X800 GTO2/XL]",
    "524C": "Rage 128 VR AGP",
    "534D": "Rage 128 4X AGP 4x",
    "5354": "Mach 64 VT",
    "5446": "Rage 128 PRO Ultra AGP 4x",
    "5452": "Rage 128 PRO Ultra4XL VR-R AGP",
    "5460": "RV370/M22 [Mobility Radeon X300]",
    "5461": "RV370/M22 [Mobility Radeon X300]",
    "514C": "R200 [Radeon 8500/8500 LE]",
    "514D": "R200 [Radeon 9100]",
    "5157": "RV200 [Radeon 7500/7500 LE]",
    "5159": "RV100 [Radeon 7000 / Radeon VE]",
    "515E": "ES1000",
    "5245": "Rage 128 GL PCI",
    "5246": "Rage 128 (Rage 4) series",
    "524B": "Rage 128 VR PCI",
    "4E6A": "RV350 [Radeon 9800 XT] (Secondary)",
    "4E71": "RV350/M10 [Mobility Radeon 9600] (Secondary)",
    "4F72": "RV250 [Radeon 9000 Series]",
    "4F73": "RV250 [Radeon 9000 Series] (Secondary)",
    "5044": "All-In-Wonder 128 PCI",
    "5046": "Rage 4 [Rage 128 PRO AGP 4X]",
    "5050": "Rage 4 [Rage 128 PRO PCI / Xpert 128 PCI]",
    "5052": "Rage 4 [Rage 128 PRO AGP 4X]",
    "5144": "R100 [Radeon 7200 / All-In-Wonder Radeon]",
    "5148": "R200 GL [FireGL 8800]",
    "4E54": "RV350/M10 GL [Mobility FireGL T2]",
    "4E56": "RV360/M12 [Mobility Radeon 9550]",
    "4E64": "R300 [Radeon 9700 PRO] (Secondary)",
    "4E65": "R300 [Radeon 9500 PRO] (Secondary)",
    "4E66": "RV350 [Radeon 9600] (Secondary)",
    "4E67": "R300 GL [FireGL X1] (Secondary)",
    "4E68": "R350 [Radeon 9800 PRO] (Secondary)",
    "4E69": "R350 [Radeon 9800] (Secondary)",
    "4E48": "R350 [Radeon 9800 Series]",
    "4E49": "R350 [Radeon 9800]",
    "4E4A": "R360 [Radeon 9800 XXL/XT]",
    "4E4B": "R350 GL [FireGL X2 AGP Pro]",
    "4E50": "RV350/M10 / RV360/M11 [Mobility Radeon 9600 (PRO) / 9700]",
    "4E51": "RV350 [Radeon 9550/9600/X1050 Series]",
    "4E52": "RV350/M10 [Mobility Radeon 9500/9700 SE]",
    "4C66": "RV250/M9 GL [Mobility FireGL 9000/Radeon 9000]",
    "4C6E": "RV250/M9 [Mobility Radeon 9000] (Secondary)",
    "4D46": "Rage Mobility 128 AGP 4X/Mobility M4",
    "4D52": "Theater 550 PRO PCI [ATI TV Wonder 550]",
    "4D53": "Theater 550 PRO PCIe",
    "4E44": "R300 [Radeon 9700/9700 PRO]",
    "4E45": "R300 [Radeon 9500 PRO/9700]",
    "4E46": "R300 [Radeon 9600 TX]",
    "4E47": "R300 GL [FireGL X1]",
    "4C4E": "Rage Mobility L AGP 2x",
    "4C46": "Rage Mobility 128 AGP 2X/Mobility M3",
    "4C47": "3D Rage IIC PCI / Mobility Radeon 7500/7500C",
    "4C49": "3D Rage LT PRO PCI",
    "4C4D": "Rage Mobility AGP 2x Series",
    "4C50": "Rage 3 LT [3D Rage LT PRO PCI]",
    "4C52": "M1 [Rage Mobility-M1 PCI]",
    "4C54": "264LT [Mach64 LT]",
    "4C57": "RV200/M7 [Mobility Radeon 7500]",
    "4C58": "RV200/M7 GL [Mobility FireGL 7800]",
    "4C59": "RV100/M6 [Rage/Radeon Mobility Series]",
    "4A6A": "R420 [Radeon X800] (Secondary)",
    "4A6B": "R420 [Radeon X800 XT AGP] (Secondary)",
    "4A70": "R420 [Radeon X800 XT Platinum Edition AGP] (Secondary)",
    "4A74": "R420 [Radeon X800 VE] (Secondary)",
    "4B49": "R481 [Radeon X850 XT AGP]",
    "4B4B": "R481 [Radeon X850 PRO AGP]",
    "4B4C": "R481 [Radeon X850 XT Platinum Edition AGP]",
    "4B69": "R481 [Radeon X850 XT AGP] (Secondary)",
    "4B6B": "R481 [Radeon X850 PRO AGP] (Secondary)",
    "4B6C": "R481 [Radeon X850 XT Platinum Edition AGP] (Secondary)",
    "4C42": "Mach64 LT [3D Rage LT PRO AGP]",
    "496E": "RV250 [Radeon 9000] (Secondary)",
    "4A49": "R420 [Radeon X800 PRO/GTO AGP]",
    "4A4A": "R420 [Radeon X800 GT AGP]",
    "4A4B": "R420 [Radeon X800 AGP Series]",
    "4A4D": "R420 GL [FireGL X3-256]",
    "4A4E": "RV420/M18 [Mobility Radeon 9800]",
    "4A4F": "R420 [Radeon X850 AGP]",
    "4A50": "R420 [Radeon X800 XT Platinum Edition AGP]",
    "4A54": "R420 [Radeon X800 VE AGP]",
    "4A69": "R420 [Radeon X800 PRO/GTO] (Secondary)",
    "4966": "RV250 [Radeon 9000 Series]",
    "4753": "Rage XC",
    "4754": "Mach64 GT/GT-B [3D Rage I/II]",
    "4755": "Mach64 GT-B [3D Rage II+ DVD]",
    "4756": "Rage 2 [3D Rage IIC PCI]",
    "4757": "Rage 2 [3D Rage IIC AGP]",
    "4758": "Mach64 GX [WinTurbo]",
    "4759": "Rage 3 [3D Rage IIC PCI]",
    "475A": "3D Rage IIC AGP",
    "4654": "Mach64 VT",
    "4742": "Rage 3 [3D Rage PRO AGP 2X]",
    "4744": "Rage 3 [3D Rage PRO AGP 1X]",
    "4749": "3D Rage PRO PCI",
    "474D": "Rage XL AGP 2X",
    "474E": "Rage XC AGP",
    "474F": "Rage XL",
    "4750": "3D Rage Pro PCI",
    "4752": "Rage 3 [Rage XL PCI]",
    "4437": "RS250 [Mobility Radeon 7000 IGP]",
    "4554": "210888ET [Mach64 ET]",
    "4630": "XENOS Parent Die (XBOX 360)",
    "4631": "XENOS Daughter Die (XBOX 360)",
}

update_modules()

if len(sys.argv) > 1 and sys.argv[1]=="--launch-uvicorn":
    launch_uvicorn()
