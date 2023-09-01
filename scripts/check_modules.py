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
from pprint import pprint

os_name = platform.system()

modules_to_check = {
    "torch": ("1.11.0", "1.13.1", "2.0.0", "2.0.1"),
    "torchvision": ("0.12.0", "0.14.1", "0.15.1", "0.15.2"),
    "sdkit": "2.0.9",
    "stable-diffusion-sdkit": "2.1.4",
    "rich": "12.6.0",
    "uvicorn": "0.19.0",
    "fastapi": "0.85.1",
    "pycloudflared": "0.2.0",
    "ruamel.yaml": "0.17.21",
    "sqlalchemy": "2.0.19",
    "python-multipart": "0.0.6",
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
        amd_gpus = setup_amd_environment()
        if module_name == "torch":
            if "Navi 3" in amd_gpus:
                # No AMD 7x00 support in rocm 5.2, needs nightly 5.5. build
                module_version = "2.1.0.dev-20230614+rocm5.5"
                index_url = "https: //download.pytorch.org/whl/nightly/rocm5.5"
            else:
                module_version = "1.13.1+rocm5.2"
        elif module_name == "torchvision":
            if "Navi 3" in amd_gpus:
                # No AMD 7x00 support in rocm 5.2, needs nightly 5.5. build
                module_version = "0.16.0.dev-20230614+rocm5.5"
                index_url = "https: //download.pytorch.org/whl/nightly/rocm5.5"
            else:
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
        if os.path.exists(f"src/{module_name}"):
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
    config_directory = os.path.dirname(__file__)  # this will be "scripts"
    config_yaml = os.path.join(config_directory, "..", "config.yaml")
    config_json = os.path.join(config_directory, "config.json")

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
    return config

def setup_amd_environment():
    gpus = list(filter(lambda x: ("amdgpu" in x), open("/proc/bus/pci/devices", "r").readlines()))
    gpus = [ x.split("\t")[1].upper() for x in gpus ]
    gpus = [ AMD_PCI_IDs[x] for x in gpus if x in AMD_PCI_IDs ]
    i=0
    supported_gpus=[]
    for gpu in gpus:
        print(f"Found AMD GPU {gpu}.")
        if gpu.startswith("Navi 1"):
            print("--- Applying Navi 1 settings")
            os.environ["HSA_OVERRIDE_GFX_VERSION"]="10.3.0"
            os.environ["FORCE_FULL_PRECISION"]="yes"
            os.environ["HIP_VISIBLE_DEVICES"]=str(i)
            supported_gpus.append("Navi 1")
        elif gpu.startswith("Navi 2"):    
            print("--- Applying Navi 2 settings")
            os.environ["HSA_OVERRIDE_GFX_VERSION"]="10.3.0"
            os.environ["HIP_VISIBLE_DEVICES"]=str(i)
            supported_gpus.append("Navi 2")
        elif gpu.startswith("Navi 3"):
            print("--- Applying Navi 3 settings")
            os.environ["HSA_OVERRIDE_GFX_VERSION"]="11.0.0"
            os.environ["HIP_VISIBLE_DEVICES"]=str(i)
            supported_gpus.append("Navi 3")
        else:
            print("--- This GPU is probably not supported by ROCm\n")
        i+=1
    return supported_gpus


def launch_uvicorn():
    config = get_config()

    # pprint(config)

    with open("scripts/install_status.txt","a") as f:
        f.write("sd_weights_downloaded\n")
        f.write("sd_install_complete\n")

    print("\n\nEasy Diffusion installation complete, starting the server!\n\n")

    os.environ["SD_PATH"] = str(Path(Path.cwd(), "stable-diffusion"))
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if os_name == "Windows":
        os.environ["PYTHONPATH"] = str(Path( os.environ["INSTALL_ENV_DIR"], "lib", "site-packages"))
    else:
        os.environ["PYTHONPATH"] = str(Path( os.environ["INSTALL_ENV_DIR"], "lib", "python3.8", "site-packages"))
    os.environ["SD_UI_PATH"] = str(Path(Path.cwd(), "ui"))

    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    print(f"Python:  {shutil.which('python')}")
    print(f"Version: {platform. python_version()}")

    bind_ip = "127.0.0.1"
    listen_port = 9000
    if "net" in config:
        print("Checking network settings")
        if "listen_port" in config["net"]:
            listen_port = config["net"]["listen_port"]
            print("Set listen port to ", listen_port)
        if "listen_to_network" in config["net"] and config["net"]["listen_to_network"] == True:
            if "bind_ip" in config["net"]:
                bind_ip = config["net"]["bind_ip"]
            else:
                bind_ip = "0.0.0.0"
            print("Set bind_ip to ", bind_ip)

    os.chdir("stable-diffusion")

    if is_amd_on_linux():
        setup_amd_environment()

    print("\nLaunching uvicorn\n")
    os.system(f'python -m uvicorn main:server_api --app-dir "{os.environ["SD_UI_PATH"]}" --port {listen_port} --host {bind_ip} --log-level error')

### Start

# This list would probably be a good candidate for an import, but since PYTHONPATH and other settings
# have not been initialized yet, I keep the list here for the moment  -- JeLuF
AMD_PCI_IDs = {
    "1002AC0C": "Theater 506A USB",
    "1002AC0D": "Theater 506A USB",
    "1002AC0E": "Theater 506A External USB",
    "1002AC0F": "Theater 506A External USB",
    "1002AC12": "Theater HD T507 (DVB-T) TV tuner/capture device",
    "1002AC02": "TV Wonder HD 600 PCIe",
    "1002AC03": "Theater 506 PCIe",
    "1002AC04": "Theater 506 USB",
    "1002AC05": "Theater 506 USB",
    "1002AC06": "Theater 506 External USB",
    "1002AC07": "Theater 506 External USB",
    "1002AC08": "Theater 506A World-Wide Analog Decoder + Demodulator",
    "1002AC09": "Theater 506A World-Wide Analog Decoder + Demodulator",
    "1002AC0A": "Theater 506A PCIe",
    "1002AC0B": "Theater 506A PCIe",
    "1002AC00": "Theater 506 World-Wide Analog Decoder",
    "1002AC01": "Theater 506 World-Wide Analog Decoder",
    "1002999D": "Richland [Radeon HD 8550D]",
    "100299A0": "Trinity 2 [Radeon HD 7520G]",
    "100299A2": "Trinity 2 [Radeon HD 7420G]",
    "100299A4": "Trinity 2 [Radeon HD 7400G]",
    "10029996": "Richland [Radeon HD 8470D]",
    "10029997": "Richland [Radeon HD 8350G]",
    "10029998": "Richland [Radeon HD 8370D]",
    "10029999": "Richland [Radeon HD 8510G]",
    "1002999A": "Richland [Radeon HD 8410G]",
    "1002999B": "Richland [Radeon HD 8310G]",
    "1002999C": "Richland [Radeon HD 8650D]",
    "10029925": "Kingston/Clayton/Jupiter/Gladius/Montego HDMI Controller",
    "10029926": "Jupiter",
    "10029990": "Trinity 2 [Radeon HD 7520G]",
    "10029991": "Trinity 2 [Radeon HD 7540D]",
    "10029992": "Trinity 2 [Radeon HD 7420G]",
    "10029993": "Trinity 2 [Radeon HD 7480D]",
    "10029994": "Trinity 2 [Radeon HD 7400G]",
    "10029995": "Richland [Radeon HD 8450G]",
    "10029917": "Trinity [Radeon HD 7620G]",
    "10029918": "Trinity [Radeon HD 7600G]",
    "10029919": "Trinity [Radeon HD 7500G]",
    "1002991E": "Bishop [Xbox One S APU]",
    "10029920": "Liverpool [Playstation 4 APU]",
    "10029922": "Starshp",
    "10029923": "Starsha2 [Kingston/Clayton]",
    "10029924": "Gladius",
    "10029909": "Trinity [Radeon HD 7500G]",
    "1002990A": "Trinity [Radeon HD 7500G]",
    "1002990B": "Richland [Radeon HD 8650G]",
    "1002990C": "Richland [Radeon HD 8670D]",
    "1002990D": "Richland [Radeon HD 8550G]",
    "1002990E": "Richland [Radeon HD 8570D]",
    "1002990F": "Richland [Radeon HD 8610G]",
    "10029910": "Trinity [Radeon HD 7660G]",
    "10029913": "Trinity [Radeon HD 7640G]",
    "10029901": "Trinity [Radeon HD 7660D]",
    "10029903": "Trinity [Radeon HD 7640G]",
    "10029904": "Trinity [Radeon HD 7560D]",
    "10029905": "Trinity GL [FirePro A300]",
    "10029906": "Trinity GL [FirePro A320]",
    "10029907": "Trinity [Radeon HD 7620G]",
    "10029908": "Trinity [Radeon HD 7600G]",
    "1002985F": "Mullins",
    "10029874": "Wani [Radeon R5/R6/R7 Graphics]",
    "10029890": "Amur",
    "100298C0": "Nolan",
    "100298E4": "Stoney [Radeon R2/R3/R4/R5 Graphics]",
    "10029900": "Trinity [Radeon HD 7660G]",
    "10029858": "Mullins",
    "10029859": "Mullins",
    "1002985A": "Mullins",
    "1002985B": "Mullins",
    "1002985C": "Mullins",
    "1002985D": "Mullins",
    "1002985E": "Mullins",
    "10029852": "Mullins [Radeon R2 Graphics]",
    "10029853": "Mullins [Radeon R2 Graphics]",
    "10029854": "Mullins [Radeon R3E Graphics]",
    "10029855": "Mullins [Radeon R6 Graphics]",
    "10029856": "Mullins [Radeon R1E/R2E Graphics]",
    "10029857": "Mullins [Radeon APU XX-2200M with R2 Graphics]",
    "10029836": "Kabini [Radeon HD 8280 / R3 Series]",
    "10029837": "Kabini [Radeon HD 8280E]",
    "10029838": "Kabini [Radeon HD 8240 / R3 Series]",
    "10029839": "Kabini [Radeon HD 8180]",
    "1002983D": "Temash [Radeon HD 8250/8280G]",
    "10029850": "Mullins [Radeon R3 Graphics]",
    "10029851": "Mullins [Radeon R4/R5 Graphics]",
    "10029808": "Wrestler [Radeon HD 7340]",
    "10029809": "Wrestler [Radeon HD 7310]",
    "1002980A": "Wrestler [Radeon HD 7290]",
    "10029830": "Kabini [Radeon HD 8400 / R3 Series]",
    "10029831": "Kabini [Radeon HD 8400E]",
    "10029832": "Kabini [Radeon HD 8330]",
    "10029833": "Kabini [Radeon HD 8330E]",
    "10029834": "Kabini [Radeon HD 8210]",
    "10029835": "Kabini [Radeon HD 8310E]",
    "10029714": "RS880 [Radeon HD 4290]",
    "10029715": "RS880 [Radeon HD 4250]",
    "10029802": "Wrestler [Radeon HD 6310]",
    "10029803": "Wrestler [Radeon HD 6310]",
    "10029804": "Wrestler [Radeon HD 6250]",
    "10029805": "Wrestler [Radeon HD 6250]",
    "10029806": "Wrestler [Radeon HD 6320]",
    "10029807": "Wrestler [Radeon HD 6290]",
    "1002964B": "Sumo",
    "1002964C": "Sumo",
    "1002964E": "Sumo",
    "1002964F": "Sumo",
    "10029710": "RS880 [Radeon HD 4200]",
    "10029712": "RS880M [Mobility Radeon HD 4225/4250]",
    "10029713": "RS880M [Mobility Radeon HD 4100]",
    "10029643": "SuperSumo [Radeon HD 6380G]",
    "10029644": "SuperSumo [Radeon HD 6410D]",
    "10029645": "SuperSumo [Radeon HD 6410D]",
    "10029647": "Sumo [Radeon HD 6520G]",
    "10029648": "Sumo [Radeon HD 6480G]",
    "10029649": "SuperSumo [Radeon HD 6480G]",
    "1002964A": "Sumo [Radeon HD 6530D]",
    "10029642": "SuperSumo [Radeon HD 6370D]",
    "10029615": "RS780E [Radeon HD 3200]",
    "10029610": "RS780 [Radeon HD 3200]",
    "10029611": "RS780C [Radeon 3100]",
    "10029612": "RS780M [Mobility Radeon HD 3200]",
    "10029613": "RS780MC [Mobility Radeon HD 3100]",
    "10029614": "RS780D [Radeon HD 3300]",
    "10029616": "RS780L [Radeon 3000]",
    "10029640": "Sumo [Radeon HD 6550D]",
    "10029641": "Sumo [Radeon HD 6620G]",
    "100295C4": "RV620/M82 [Mobility Radeon HD 3450/3470]",
    "100295C5": "RV620 LE [Radeon HD 3450]",
    "100295C6": "RV620 LE [Radeon HD 3450 AGP]",
    "100295C9": "RV620 LE [Radeon HD 3450 PCI]",
    "100295CC": "RV620 GL [FirePro V3700]",
    "100295CD": "RV620 GL [FirePro 2450]",
    "100295CF": "RV620 GL [FirePro 2260]",
    "10029591": "RV635/M86 [Mobility Radeon HD 3650]",
    "10029593": "RV635/M86 [Mobility Radeon HD 3670]",
    "10029595": "RV635/M86 GL [Mobility FireGL V5700]",
    "10029596": "RV635 PRO [Radeon HD 3650 AGP]",
    "10029597": "RV635 PRO [Radeon HD 3650 AGP]",
    "10029598": "RV635 [Radeon HD 3650/3750/4570/4580]",
    "10029599": "RV635 PRO [Radeon HD 3650 AGP]",
    "100295C0": "RV620 PRO [Radeon HD 3470]",
    "100295C2": "RV620/M82 [Mobility Radeon HD 3410/3430]",
    "10029586": "RV630 XT [Radeon HD 2600 XT AGP]",
    "10029587": "RV630 PRO [Radeon HD 2600 PRO AGP]",
    "10029588": "RV630 XT [Radeon HD 2600 XT]",
    "10029589": "RV630 PRO [Radeon HD 2600 PRO]",
    "1002958A": "RV630 [Radeon HD 2600 X2]",
    "1002958B": "RV630/M76 [Mobility Radeon HD 2600 XT]",
    "1002958C": "RV630 GL [FireGL V5600]",
    "1002958D": "RV630 GL [FireGL V3600]",
    "1002954F": "RV710 [Radeon HD 4350/4550]",
    "10029552": "RV710/M92 [Mobility Radeon HD 4330/4350/4550]",
    "10029553": "RV710/M92 [Mobility Radeon HD 4530/4570/5145/530v/540v/545v]",
    "10029555": "RV711/M93 [Mobility Radeon HD 4350/4550/530v/540v/545v / FirePro RG220]",
    "10029557": "RV711/M93 GL [FirePro RG220]",
    "1002955F": "RV710/M92 [Mobility Radeon HD 4330]",
    "10029580": "RV630 [Radeon HD 2600 PRO]",
    "10029581": "RV630/M76 [Mobility Radeon HD 2600]",
    "10029583": "RV630/M76 [Mobility Radeon HD 2600 XT/2700]",
    "1002950F": "R680 [Radeon HD 3870 X2]",
    "10029511": "RV670 GL [FireGL V7700]",
    "10029513": "RV670 [Radeon HD 3850 X2]",
    "10029515": "RV670 PRO [Radeon HD 3850 AGP]",
    "10029519": "RV670 GL [FireStream 9170]",
    "10029540": "RV710 [Radeon HD 4550]",
    "10029500": "RV670 [Radeon HD 3850 X2]",
    "10029501": "RV670 [Radeon HD 3870]",
    "10029504": "RV670/M88 [Mobility Radeon HD 3850]",
    "10029505": "RV670 [Radeon HD 3690/3850]",
    "10029506": "RV670/M88 [Mobility Radeon HD 3850 X2]",
    "10029507": "RV670 [Radeon HD 3830]",
    "10029508": "RV670/M88-XT [Mobility Radeon HD 3870]",
    "10029509": "RV670/M88 [Mobility Radeon HD 3870 X2]",
    "100294C1": "RV610 [Radeon HD 2400 PRO/XT]",
    "100294C3": "RV610 [Radeon HD 2400 PRO]",
    "100294C4": "RV610 LE [Radeon HD 2400 PRO AGP]",
    "100294C5": "RV610 [Radeon HD 2400 LE]",
    "100294C7": "RV610 [Radeon HD 2350]",
    "100294C8": "RV610/M74 [Mobility Radeon HD 2400 XT]",
    "100294C9": "RV610/M72-S [Mobility Radeon HD 2400]",
    "100294CB": "RV610 [Radeon E2400]",
    "100294CC": "RV610 LE [Radeon HD 2400 PRO PCI]",
    "10029498": "RV730 PRO [Radeon HD 4650]",
    "1002949C": "RV730 GL [FirePro V7750]",
    "1002949E": "RV730 GL [FirePro V5700]",
    "1002949F": "RV730 GL [FirePro V3750]",
    "100294A0": "RV740/M97 [Mobility Radeon HD 4830]",
    "100294A1": "RV740/M97-XT [Mobility Radeon HD 4860]",
    "100294A3": "RV740/M97 GL [FirePro M7740]",
    "100294B3": "RV740 PRO [Radeon HD 4770]",
    "100294B4": "RV740 PRO [Radeon HD 4750]",
    "10029462": "RV790 [Radeon HD 4860]",
    "1002946A": "RV770 GL [FirePro M7750]",
    "10029480": "RV730/M96 [Mobility Radeon HD 4650/5165]",
    "10029488": "RV730/M96-XT [Mobility Radeon HD 4670]",
    "10029489": "RV730/M96 GL [Mobility FireGL V5725]",
    "10029490": "RV730 XT [Radeon HD 4670]",
    "10029491": "RV730/M96-CSP [Radeon E4690]",
    "10029495": "RV730 [Radeon HD 4600 AGP Series]",
    "1002944A": "RV770/M98L [Mobility Radeon HD 4850]",
    "1002944B": "RV770/M98 [Mobility Radeon HD 4850 X2]",
    "1002944C": "RV770 LE [Radeon HD 4830]",
    "1002944E": "RV770 CE [Radeon HD 4710]",
    "10029450": "RV770 GL [FireStream 9270]",
    "10029452": "RV770 GL [FireStream 9250]",
    "10029456": "RV770 GL [FirePro V8700]",
    "1002945A": "RV770/M98-XT [Mobility Radeon HD 4870]",
    "10029460": "RV790 [Radeon HD 4890]",
    "1002940A": "R600 GL [FireGL V8650]",
    "1002940B": "R600 GL [FireGL V8600]",
    "1002940F": "R600 GL [FireGL V7600]",
    "10029440": "RV770 [Radeon HD 4870]",
    "10029441": "R700 [Radeon HD 4870 X2]",
    "10029442": "RV770 [Radeon HD 4850]",
    "10029443": "R700 [Radeon HD 4850 X2]",
    "10029444": "RV770 GL [FirePro V8750]",
    "10029446": "RV770 GL [FirePro V7760]",
    "1002793F": "RS690M [Radeon Xpress 1200/1250/1270] (Secondary)",
    "10027941": "RS600 [Radeon Xpress 1250]",
    "10027942": "RS600M [Radeon Xpress 1250]",
    "1002796E": "RS740 [Radeon 2100]",
    "10029400": "R600 [Radeon HD 2900 PRO/XT]",
    "10029401": "R600 [Radeon HD 2900 XT]",
    "10029403": "R600 [Radeon HD 2900 PRO]",
    "10029405": "R600 [Radeon HD 2900 GT]",
    "1002791E": "RS690 [Radeon X1200]",
    "1002791F": "RS690M [Radeon Xpress 1200/1250/1270]",
    "10027835": "RS350M [Mobility Radeon 9000 IGP]",
    "10027448": "Navi 31 [Radeon Pro W7900]",
    "1002744C": "Navi 31 [Radeon RX 7900 XT/7900 XTX]",
    "1002745E": "Navi 31 [Radeon Pro W7800]",
    "10027480": "Navi 33 [Radeon RX 7700S/7600S/7600M XT]",
    "10027483": "Navi 33 [Radeon RX 7600M/7600M XT]",
    "10027489": "Navi 33",
    "10027834": "RS350 [Radeon 9100 PRO/XT IGP]",
    "10027446": "Navi 31 USB",
    "1002743F": "Navi 24 [Radeon RX 6400/6500 XT/6500M]",
    "10027421": "Navi 24 [Radeon PRO W6500M]",
    "10027422": "Navi 24 [Radeon PRO W6400]",
    "10027423": "Navi 24 [Radeon PRO W6300/W6300M]",
    "10027424": "Navi 24 [Radeon RX 6300]",
    "100273F0": "Navi 33 [Radeon RX 7600M XT]",
    "100273EF": "Navi 23 [Radeon RX 6650 XT / 6700S / 6800S]",
    "100273E1": "Navi 23 WKS-XM [Radeon PRO W6600M]",
    "100273FF": "Navi 23 [Radeon RX 6600/6600 XT/6600M]",
    "100273E3": "Navi 23 WKS-XL [Radeon PRO W6600]",
    "100273E4": "Navi 23 USB",
    "10027408": "Aldebaran/MI200 [Instinct MI250X]",
    "1002740C": "Aldebaran/MI200 [Instinct MI250X/MI250]",
    "1002740F": "Aldebaran/MI200 [Instinct MI210]",
    "100273CE": "Navi 22-XL SRIOV MxGPU",
    "100273AF": "Navi 21 [Radeon RX 6900 XT]",
    "100273BF": "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]",
    "100273C4": "Navi 22 USB",
    "100273C3": "Navi 22",
    "100273E0": "Navi 23",
    "100273DF": "Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]",
    "100273A5": "Navi 21 [Radeon RX 6950 XT]",
    "100273AE": "Navi 21 [Radeon Pro V620 MxGPU]",
    "100273A2": "Navi 21 Pro-XTA [Radeon Pro W6900X]",
    "100273A3": "Navi 21 GL-XL [Radeon PRO W6800]",
    "100273A4": "Navi 21 USB",
    "100273AB": "Navi 21 Pro-XLA [Radeon Pro W6800X/Radeon Pro W6800X Duo]",
    "1002734F": "Navi 14 [Radeon Pro W5300M]",
    "10027360": "Navi 12 [Radeon Pro 5600M/V520/BC-160]",
    "100273A1": "Navi 21 [Radeon Pro V620]",
    "10027362": "Navi 12 [Radeon Pro V520/V540]",
    "10027388": "Arcturus GL-XL",
    "1002738C": "Arcturus GL-XL [Instinct MI100]",
    "1002738E": "Arcturus GL-XL [Instinct MI100]",
    "10027319": "Navi 10 [Radeon Pro 5700 XT]",
    "1002731B": "Navi 10 [Radeon Pro 5700]",
    "1002731F": "Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT]",
    "10027340": "Navi 14 [Radeon RX 5500/5500M / Pro 5500M]",
    "10027341": "Navi 14 [Radeon Pro W5500]",
    "10027347": "Navi 14 [Radeon Pro W5500M]",
    "1002731E": "TDC-150",
    "100272A0": "RV570 [Radeon X1950 PRO] (Secondary)",
    "100272A8": "RV570 [Radeon X1950 GT] (Secondary)",
    "100272B1": "RV560 [Radeon X1650 XT] (Secondary)",
    "100272B3": "RV560 [Radeon X1650 GT] (Secondary)",
    "10027300": "Fiji [Radeon R9 FURY / NANO Series]",
    "10027310": "Navi 10 [Radeon Pro W5700X]",
    "10027312": "Navi 10 [Radeon Pro W5700]",
    "10027314": "Navi 10 USB",
    "1002724E": "R580 GL [FireGL V7350]",
    "10027269": "R580 [Radeon X1900 XT] (Secondary)",
    "1002726B": "R580 [Radeon X1900 GT] (Secondary)",
    "1002726E": "R580 [AMD Stream Processor] (Secondary)",
    "10027280": "RV570 [Radeon X1950 PRO]",
    "10027288": "RV570 [Radeon X1950 GT]",
    "10027291": "RV560 [Radeon X1650 XT]",
    "10027293": "RV560 [Radeon X1650 GT]",
    "100271F2": "RV530 GL [FireGL V3400] (Secondary)",
    "10027210": "RV550/M71 [Mobility Radeon HD 2300]",
    "10027211": "RV550/M71 [Mobility Radeon X2300 HD]",
    "10027240": "R580+ [Radeon X1950 XTX]",
    "10027244": "R580+ [Radeon X1950 XT]",
    "10027248": "R580 [Radeon X1950]",
    "10027249": "R580 [Radeon X1900 XT]",
    "1002724B": "R580 [Radeon X1900 GT]",
    "100271D6": "RV530/M66-XT [Mobility Radeon X1700]",
    "100271DE": "RV530/M66 [Mobility Radeon X1700/X2500]",
    "100271E0": "RV530 [Radeon X1600] (Secondary)",
    "100271E1": "RV535 [Radeon X1650 PRO] (Secondary)",
    "100271E2": "RV530 [Radeon X1600] (Secondary)",
    "100271E6": "RV530 [Radeon X1650] (Secondary)",
    "100271E7": "RV535 [Radeon X1650 PRO] (Secondary)",
    "100271C4": "RV530/M56 GL [Mobility FireGL V5200]",
    "100271C5": "RV530/M56-P [Mobility Radeon X1600]",
    "100271C6": "RV530LE [Radeon X1600/X1650 PRO]",
    "100271C7": "RV535 [Radeon X1650 PRO]",
    "100271CE": "RV530 [Radeon X1300 XT/X1600 PRO]",
    "100271D2": "RV530 GL [FireGL V3400]",
    "100271D4": "RV530/M66 GL [Mobility FireGL V5250]",
    "100271D5": "RV530/M66-P [Mobility Radeon X1700]",
    "100271A7": "RV516 [Radeon X1300/X1550 Series] (Secondary)",
    "100271BB": "RV516 GL [FireMV 2250] (Secondary)",
    "100271C0": "RV530 [Radeon X1600 XT/X1650 GTO]",
    "100271C1": "RV535 [Radeon X1650 PRO]",
    "100271C2": "RV530 [Radeon X1600 PRO]",
    "100271C3": "RV530 [Radeon X1600 PRO]",
    "1002718D": "RV516/M64-CSP128 [Mobility Radeon X1450]",
    "10027193": "RV516 [Radeon X1550 Series]",
    "10027196": "RV516/M62-S [Mobility Radeon X1350]",
    "1002719B": "RV516 GL [FireMV 2250]",
    "1002719F": "RV516 [Radeon X1550 Series]",
    "100271A0": "RV516 [Radeon X1300/X1550 Series] (Secondary)",
    "100271A1": "RV516 [Radeon X1600/X1650 Series] (Secondary)",
    "100271A3": "RV516 [Radeon X1300/X1550 Series] (Secondary)",
    "10027186": "RV516/M64 [Mobility Radeon X1450]",
    "10027187": "RV516 [Radeon X1300/X1550 Series]",
    "10027188": "RV516/M64-S [Mobility Radeon X2300]",
    "1002718A": "RV516/M64 [Mobility Radeon X2300]",
    "1002718B": "RV516/M62 [Mobility Radeon X1350]",
    "1002718C": "RV516/M62-CSP64 [Mobility Radeon X1350]",
    "10027162": "RV515 PRO [Radeon X1300/X1550 Series] (Secondary)",
    "10027163": "RV505 [Radeon X1550 Series] (Secondary)",
    "10027166": "RV515 [Radeon X1300/X1550 Series] (Secondary)",
    "10027167": "RV515 [Radeon X1550 64-bit] (Secondary)",
    "10027172": "RV515 GL [FireGL V3300] (Secondary)",
    "10027173": "RV515 GL [FireGL V3350] (Secondary)",
    "10027181": "RV516 [Radeon X1600/X1650 Series]",
    "10027183": "RV516 [Radeon X1300/X1550 Series]",
    "10027145": "RV515/M54 [Mobility Radeon X1400]",
    "10027146": "RV515 [Radeon X1300/X1550]",
    "10027147": "RV505 [Radeon X1550 64-bit]",
    "10027149": "RV515/M52 [Mobility Radeon X1300]",
    "1002714A": "RV515/M52 [Mobility Radeon X1300]",
    "10027152": "RV515 GL [FireGL V3300]",
    "10027153": "RV515 GL [FireGL V3350]",
    "1002715F": "RV505 CE [Radeon X1550 64-bit]",
    "10027120": "R520 [Radeon X1800] (Secondary)",
    "10027124": "R520 GL [FireGL V7200] (Secondary)",
    "10027129": "R520 [Radeon X1800] (Secondary)",
    "1002712E": "R520 GL [FireGL V7300] (Secondary)",
    "1002712F": "R520 GL [FireGL V7350] (Secondary)",
    "10027140": "RV515 [Radeon X1300/X1550/X1600 Series]",
    "10027142": "RV515 PRO [Radeon X1300/X1550 Series]",
    "10027143": "RV505 [Radeon X1300/X1550 Series]",
    "10027102": "R520/M58 [Mobility Radeon X1800]",
    "10027104": "R520 GL [FireGL V7200 / Barco MXTR-5100]",
    "10027109": "R520 [Radeon X1800 XL]",
    "1002710A": "R520 [Radeon X1800 GTO]",
    "1002710B": "R520 [Radeon X1800 GTO]",
    "1002710E": "R520 GL [FireGL V7300]",
    "1002710F": "R520 GL [FireGL V7350]",
    "100269A1": "Vega 12",
    "100269A2": "Vega 12",
    "100269A3": "Vega 12",
    "100269AF": "Vega 12 [Radeon Pro Vega 20]",
    "10026FDF": "Polaris 20 XL [Radeon RX 580 2048SP]",
    "10027100": "R520 [Radeon X1800 XT]",
    "10027101": "R520/M58 [Mobility Radeon X1800 XT]",
    "10026980": "Polaris12",
    "10026981": "Lexa XT [Radeon PRO WX 3200]",
    "10026985": "Lexa XT [Radeon PRO WX 3100]",
    "10026986": "Polaris12",
    "10026987": "Lexa [Radeon 540X/550X/630 / RX 640 / E9171 MCM]",
    "10026995": "Lexa XT [Radeon PRO WX 2100]",
    "1002699F": "Lexa PRO [Radeon 540/540X/550/550X / RX 540X/550/550X]",
    "100269A0": "Vega 12",
    "1002698F": "Lexa XT [Radeon PRO WX 3100 / Barco MXRT 4700]",
    "1002692F": "Tonga XTV GL [FirePro S7150V]",
    "10026938": "Tonga XT / Amethyst XT [Radeon R9 380X / R9 M295X]",
    "10026939": "Tonga PRO [Radeon R9 285/380]",
    "1002694C": "Polaris 22 XT [Radeon RX Vega M GH]",
    "1002694E": "Polaris 22 XL [Radeon RX Vega M GL]",
    "1002694F": "Polaris 22 MGL XL [Radeon Pro WX Vega M GL]",
    "10026930": "Tonga PRO [Radeon R9 380 4GB]",
    "1002693B": "Tonga PRO GL [FirePro W7100 / Barco MXRT-7600]",
    "100268FE": "Cedar LE",
    "10026900": "Topaz XT [Radeon R7 M260/M265 / M340/M360 / M440/M445 / 530/535 / 620/625 Mobile]",
    "10026901": "Topaz PRO [Radeon R5 M255]",
    "10026907": "Meso XT [Radeon R5 M315]",
    "10026920": "Amethyst [Radeon R9 M395/ M395X Mac Edition]",
    "10026921": "Amethyst XT [Radeon R9 M295X / M390X]",
    "10026929": "Tonga XT GL [FirePro S7150]",
    "1002692B": "Tonga PRO GL [FirePro W7100]",
    "100268FA": "Cedar [Radeon HD 7350/8350 / R5 220]",
    "100268E4": "Robson CE [Radeon HD 6370M/7370M]",
    "100268E5": "Robson LE [Radeon HD 6330M]",
    "100268E8": "Cedar",
    "100268E9": "Cedar [ATI FirePro (FireGL) Graphics Adapter]",
    "100268F1": "Cedar GL [FirePro 2460]",
    "100268F2": "Cedar GL [FirePro 2270]",
    "100268F8": "Cedar [Radeon HD 7300 Series]",
    "100268F9": "Cedar [Radeon HD 5000/6000/7350/8350 Series]",
    "100268C8": "Redwood XT GL [FirePro V4800]",
    "100268C9": "Redwood PRO GL [FirePro V3800]",
    "100268D8": "Redwood XT [Radeon HD 5670/5690/5730]",
    "100268D9": "Redwood PRO [Radeon HD 5550/5570/5630/6510/6610/7570]",
    "100268DA": "Redwood LE [Radeon HD 5550/5570/5630/6390/6490/7570]",
    "100268DE": "Redwood",
    "100268E0": "Park [Mobility Radeon HD 5430/5450/5470]",
    "100268E1": "Park [Mobility Radeon HD 5430]",
    "100268A9": "Juniper XT [FirePro V5800]",
    "100268B8": "Juniper XT [Radeon HD 5770]",
    "100268B9": "Juniper LE [Radeon HD 5670 640SP Edition]",
    "100268BA": "Juniper XT [Radeon HD 6770]",
    "100268BE": "Juniper PRO [Radeon HD 5750]",
    "100268BF": "Juniper PRO [Radeon HD 6750]",
    "100268C0": "Madison [Mobility Radeon HD 5730 / 6570M]",
    "100268C1": "Madison [Mobility Radeon HD 5650/5750 / 6530M/6550M]",
    "100268C7": "Pinewood [Mobility Radeon HD 5570/6550A]",
    "10026898": "Cypress XT [Radeon HD 5870]",
    "10026899": "Cypress PRO [Radeon HD 5850]",
    "1002689B": "Cypress PRO [Radeon HD 6800 Series]",
    "1002689C": "Hemlock [Radeon HD 5970]",
    "1002689D": "Hemlock [Radeon HD 5970]",
    "1002689E": "Cypress LE [Radeon HD 5830]",
    "100268A0": "Broadway XT [Mobility Radeon HD 5870]",
    "100268A1": "Broadway PRO [Mobility Radeon HD 5850]",
    "100268A8": "Granville [Radeon HD 6850M/6870M]",
    "1002686D": "Vega 10 GLXTA",
    "1002686E": "Vega 10 GLXLA",
    "1002687F": "Vega 10 XL/XT [Radeon RX Vega 56/64]",
    "10026880": "Lexington [Radeon HD 6550M]",
    "10026888": "Cypress XT [FirePro V8800]",
    "10026889": "Cypress PRO [FirePro V7800]",
    "1002688A": "Cypress XT [FirePro V9800]",
    "1002688C": "Cypress XT GL [FireStream 9370]",
    "1002688D": "Cypress PRO GL [FireStream 9350]",
    "10026869": "Vega 10 XGA [Radeon Pro Vega 48]",
    "1002686A": "Vega 10 LEA",
    "1002686B": "Vega 10 XTXA [Radeon Pro Vega 64X]",
    "10026862": "Vega 10 XT [Radeon PRO SSG]",
    "10026863": "Vega 10 XTX [Radeon Vega Frontier Edition]",
    "10026864": "Vega 10 [Radeon Pro V340/Instinct MI25x2]",
    "10026867": "Vega 10 XL [Radeon Pro Vega 56]",
    "10026868": "Vega 10 [Radeon PRO WX 8100/8200]",
    "1002686C": "Vega 10 [Instinct MI25 MxGPU/MI25x2 MxGPU/V340 MxGPU/V340L MxGPU]",
    "1002683F": "Cape Verde PRO [Radeon HD 7750/8740 / R7 250E]",
    "10026840": "Thames [Radeon HD 7500M/7600M Series]",
    "10026841": "Thames [Radeon HD 7550M/7570M/7650M]",
    "10026842": "Thames LE [Radeon HD 7000M Series]",
    "10026843": "Thames [Radeon HD 7670M]",
    "10026860": "Vega 10 [Instinct MI25/MI25x2/V340/V320]",
    "10026861": "Vega 10 XT [Radeon PRO WX 9100]",
    "10026828": "Cape Verde PRO [FirePro W600]",
    "10026829": "Cape Verde",
    "1002682A": "Venus PRO",
    "1002682B": "Cape Verde PRO / Venus LE / Tropo PRO-L [Radeon HD 8830M / R7 250 / R7 M465X]",
    "1002682C": "Cape Verde GL [FirePro W4100]",
    "1002682D": "Chelsea XT GL [FirePro M4000]",
    "1002682F": "Chelsea LP [Radeon HD 7730M]",
    "10026835": "Cape Verde PRX [Radeon R9 255 OEM]",
    "10026837": "Cape Verde LE [Radeon HD 7730/8730]",
    "1002683D": "Cape Verde XT [Radeon HD 7770/8760 / R7 250X]",
    "10026820": "Venus XTX [Radeon HD 8890M / R9 M275X/M375X]",
    "10026821": "Venus XT [Radeon HD 8870M / R9 M270X/M370X]",
    "10026822": "Venus PRO [Radeon E8860]",
    "10026823": "Venus PRO [Radeon HD 8850M / R9 M265X]",
    "10026825": "Heathrow XT [Radeon HD 7870M]",
    "10026826": "Chelsea LP [Radeon HD 7700M Series]",
    "10026827": "Heathrow PRO [Radeon HD 7850M/8850M]",
    "10026808": "Pitcairn XT GL [FirePro W7000]",
    "10026809": "Pitcairn LE GL [FirePro W5000]",
    "10026810": "Curacao XT / Trinidad XT [Radeon R7 370 / R9 270X/370X]",
    "10026811": "Curacao PRO [Radeon R7 370 / R9 270/370 OEM]",
    "10026816": "Pitcairn",
    "10026817": "Pitcairn",
    "10026818": "Pitcairn XT [Radeon HD 7870 GHz Edition]",
    "10026819": "Pitcairn PRO [Radeon HD 7850 / R7 265 / R9 270 1024SP]",
    "100267EB": "Baffin [Radeon Pro V5300X]",
    "100267EF": "Baffin [Radeon RX 460/560D / Pro 450/455/460/555/555X/560/560X]",
    "100267FF": "Baffin [Radeon RX 550 640SP / RX 560/560X]",
    "10026800": "Wimbledon XT [Radeon HD 7970M]",
    "10026801": "Neptune XT [Radeon HD 8970M]",
    "10026802": "Wimbledon",
    "10026806": "Neptune",
    "100267DF": "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]",
    "100267E0": "Baffin [Radeon Pro WX 4170]",
    "100267E1": "Baffin [Polaris11]",
    "100267E3": "Baffin [Radeon Pro WX 4100]",
    "100267E8": "Baffin [Radeon Pro WX 4130/4150]",
    "100267E9": "Baffin [Polaris11]",
    "100267D7": "Ellesmere [Radeon Pro WX 5100 / Barco MXRT-6700]",
    "100267C7": "Ellesmere [Radeon Pro WX 5100]",
    "100267CA": "Ellesmere [Polaris10]",
    "100267CC": "Ellesmere [Polaris10]",
    "100267CF": "Ellesmere [Polaris10]",
    "100267D0": "Ellesmere [Radeon Pro V7300X / V7350x2]",
    "100267D4": "Ellesmere [Radeon Pro WX 7100 / Barco MXRT-8700]",
    "100267B9": "Vesuvius [Radeon R9 295X2]",
    "100267BE": "Hawaii LE",
    "100267C0": "Ellesmere [Radeon Pro WX 7100 Mobile]",
    "100267C2": "Ellesmere [Radeon Pro V7300X / V7350x2]",
    "100267C4": "Ellesmere [Radeon Pro WX 7100]",
    "100267B8": "Hawaii XT [Radeon R9 290X Engineering Sample]",
    "100267A2": "Hawaii GL",
    "100267A8": "Hawaii",
    "100267A9": "Hawaii",
    "100267AA": "Hawaii",
    "100267B0": "Hawaii XT / Grenada XT [Radeon R9 290X/390X]",
    "100267B1": "Hawaii PRO [Radeon R9 290/390]",
    "1002679A": "Tahiti PRO [Radeon HD 7950/8950 OEM / R9 280]",
    "1002679B": "Malta [Radeon HD 7990/8990 OEM]",
    "1002679E": "Tahiti LE [Radeon HD 7870 XT]",
    "1002679F": "Tahiti",
    "100267A0": "Hawaii XT GL [FirePro W9100]",
    "100267A1": "Hawaii PRO GL [FirePro W8100]",
    "10026780": "Tahiti XT GL [FirePro W9000]",
    "10026784": "Tahiti [FirePro Series Graphics Adapter]",
    "10026788": "Tahiti [FirePro Series Graphics Adapter]",
    "1002678A": "Tahiti PRO GL [FirePro Series]",
    "10026798": "Tahiti XT [Radeon HD 7970/8970 OEM / R9 280X]",
    "10026770": "Caicos [Radeon HD 6450A/7450A]",
    "10026771": "Caicos XTX [Radeon HD 8490 / R5 235X OEM]",
    "10026772": "Caicos [Radeon HD 7450A]",
    "10026778": "Caicos XT [Radeon HD 7470/8470 / R5 235/310 OEM]",
    "10026779": "Caicos [Radeon HD 6450/7450/8450 / R5 230 OEM]",
    "1002677B": "Caicos PRO [Radeon HD 7450]",
    "10026761": "Seymour LP [Radeon HD 6430M]",
    "10026763": "Seymour [Radeon E6460]",
    "10026764": "Seymour [Radeon HD 6400M Series]",
    "10026765": "Seymour [Radeon HD 6400M Series]",
    "10026766": "Caicos",
    "10026767": "Caicos",
    "10026768": "Caicos",
    "10026759": "Turks PRO [Radeon HD 6570/7570/8550 / R5 230]",
    "1002675B": "Turks [Radeon HD 7600 Series]",
    "1002675D": "Turks PRO [Radeon HD 7570]",
    "1002675F": "Turks LE [Radeon HD 5570/6510/7510/8510]",
    "10026760": "Seymour [Radeon HD 6400M/7400M Series]",
    "10026743": "Whistler [Radeon E6760]",
    "10026749": "Turks GL [FirePro V4900]",
    "1002674A": "Turks GL [FirePro V3900]",
    "10026750": "Onega [Radeon HD 6650A/7650A]",
    "10026751": "Turks [Radeon HD 7650A/7670A]",
    "10026758": "Turks XT [Radeon HD 6670/7670]",
    "1002671F": "Cayman CE [Radeon HD 6930]",
    "10026720": "Blackcomb [Radeon HD 6970M/6990M]",
    "10026738": "Barts XT [Radeon HD 6870]",
    "10026739": "Barts PRO [Radeon HD 6850]",
    "1002673E": "Barts LE [Radeon HD 6790]",
    "10026740": "Whistler [Radeon HD 6730M/6770M/7690M XT]",
    "10026741": "Whistler [Radeon HD 6630M/6650M/6750M/7670M/7690M]",
    "10026742": "Whistler LE [Radeon HD 6610M/7610M]",
    "10026707": "Cayman LE GL [FirePro V5900]",
    "10026718": "Cayman XT [Radeon HD 6970]",
    "10026719": "Cayman PRO [Radeon HD 6950]",
    "1002671C": "Antilles [Radeon HD 6990]",
    "1002671D": "Antilles [Radeon HD 6990]",
    "100266A2": "Vega 20",
    "100266A3": "Vega 20 [Radeon Pro Vega II/Radeon Pro Vega II Duo]",
    "100266A7": "Vega 20 [Radeon Pro Vega 20]",
    "100266AF": "Vega 20 [Radeon VII]",
    "10026704": "Cayman PRO GL [FirePro V7900]",
    "10026665": "Jet PRO [Radeon R5 M230 / R7 M260DX / Radeon 520/610 Mobile]",
    "10026667": "Jet ULT [Radeon R5 M230]",
    "1002666F": "Sun LE [Radeon HD 8550M / R5 M230]",
    "100266A0": "Vega 20 [Radeon Pro/Radeon Instinct]",
    "100266A1": "Vega 20 [Radeon Pro VII/Radeon Instinct MI50 32GB]",
    "10026658": "Bonaire XTX [Radeon R7 260X/360]",
    "1002665C": "Bonaire XT [Radeon HD 7790/8770 / R7 360 / R9 260/360 OEM]",
    "1002665D": "Bonaire [Radeon R7 200 Series]",
    "1002665F": "Tobago PRO [Radeon R7 360 / R9 360 OEM]",
    "10026660": "Sun XT [Radeon HD 8670A/8670M/8690M / R5 M330 / M430 / Radeon 520 Mobile]",
    "10026663": "Sun PRO [Radeon HD 8570A/8570M]",
    "10026664": "Jet XT [Radeon R5 M240]",
    "10026631": "Oland",
    "10026640": "Saturn XT [FirePro M6100]",
    "10026641": "Saturn PRO [Radeon HD 8930M]",
    "10026646": "Bonaire XT [Radeon R9 M280X / FirePro W6150M]",
    "10026647": "Saturn PRO/XT [Radeon R9 M270X/M280X]",
    "10026649": "Bonaire [FirePro W5100]",
    "10026650": "Bonaire",
    "10026651": "Bonaire",
    "1002664D": "Bonaire [FirePro W5100 / Barco MXRT-5600]",
    "10026606": "Mars XTX [Radeon HD 8790M]",
    "10026607": "Mars LE [Radeon HD 8530M / R5 M240]",
    "10026608": "Oland GL [FirePro W2100]",
    "10026610": "Oland XT [Radeon HD 8670 / R5 340X OEM / R7 250/350/350X OEM]",
    "10026611": "Oland [Radeon HD 8570 / R5 430 OEM / R7 240/340 / Radeon 520 OEM]",
    "10026613": "Oland PRO [Radeon R7 240/340 / Radeon 520]",
    "10026617": "Oland LE [Radeon R7 240]",
    "10026609": "Oland GL [FirePro W2100 / Barco MXRT 2600]",
    "10025E4F": "RV410 [Radeon X700]",
    "10025E6B": "RV410 [Radeon X700 PRO] (Secondary)",
    "10025E6D": "RV410 [Radeon X700] (Secondary)",
    "10025F57": "R423 [Radeon X800 XT]",
    "10026600": "Mars [Radeon HD 8670A/8670M/8750M / R7 M370]",
    "10026601": "Mars [Radeon HD 8730M]",
    "10026604": "Opal XT [Radeon R7 M265/M365X/M465]",
    "10026605": "Opal PRO [Radeon R7 M260X]",
    "10025D72": "R480 [Radeon X850 XT] (Secondary)",
    "10025D77": "R423 [Radeon X800 XT] (Secondary)",
    "10025E48": "RV410 GL [FireGL V5000]",
    "10025E49": "RV410 [Radeon X700 Series]",
    "10025E4A": "RV410 [Radeon X700 XT]",
    "10025E4B": "RV410 [Radeon X700 PRO]",
    "10025E4C": "RV410 [Radeon X700 SE]",
    "10025E4D": "RV410 [Radeon X700]",
    "10025D4D": "R480 [Radeon X850 XT Platinum Edition]",
    "10025D4E": "R480 [Radeon X850 SE]",
    "10025D4F": "R480 [Radeon X800 GTO]",
    "10025D50": "R480 GL [FireGL V7200]",
    "10025D52": "R480 [Radeon X850 XT]",
    "10025D57": "R423 [Radeon X800 XT]",
    "10025D6D": "R480 [Radeon X850 XT Platinum Edition] (Secondary)",
    "10025D6F": "R480 [Radeon X800 GTO] (Secondary)",
    "10025B73": "RV370 [Radeon X300/X550/X1050 Series] (Secondary)",
    "10025B74": "RV370 GL [FireGL V3100] (Secondary)",
    "10025B75": "RV370 GL [FireMV 2200] (Secondary)",
    "10025C61": "RV280/M9+ [Mobility Radeon 9200 AGP]",
    "10025C63": "RV280/M9+ [Mobility Radeon 9200 AGP]",
    "10025D44": "RV280 [Radeon 9200 SE] (Secondary)",
    "10025D45": "RV280 GL [FireMV 2200 PCI] (Secondary)",
    "10025D48": "R423/M28 [Mobility Radeon X800 XT]",
    "10025D49": "R423/M28 GL [Mobility FireGL V5100]",
    "10025D4A": "R423/M28 [Mobility Radeon X800]",
    "10025A62": "RC410M [Mobility Radeon Xpress 200M]",
    "10025B60": "RV370 [Radeon X300]",
    "10025B62": "RV370 [Radeon X600/X600 SE]",
    "10025B63": "RV370 [Radeon X300/X550/X1050 Series]",
    "10025B64": "RV370 GL [FireGL V3100]",
    "10025B65": "RV370 GL [FireMV 2200]",
    "10025B66": "RV370X",
    "10025B70": "RV370 [Radeon X300 SE]",
    "10025B72": "RV380 [Radeon X300/X550/X1050 Series] (Secondary)",
    "10025A36": "RC4xx/RS4xx PCI Express Port 1",
    "10025A37": "RC4xx/RS4xx PCI Express Port 2",
    "10025A38": "RC4xx/RS4xx PCI Express Port 3",
    "10025A39": "RC4xx/RS4xx PCI Express Port 4",
    "10025A41": "RS400 [Radeon Xpress 200]",
    "10025A42": "RS400M [Radeon Xpress 200M]",
    "10025A61": "RC410 [Radeon Xpress 200/1100]",
    "10025A23": "RD890S/RD990 I/O Memory Management Unit (IOMMU)",
    "10025A34": "RS4xx PCI Express Port [ext gfx]",
    "10025965": "RV280 GL [FireMV 2200 PCI]",
    "10025974": "RS482/RS485 [Radeon Xpress 1100/1150]",
    "10025975": "RS482M [Mobility Radeon Xpress 200]",
    "10025960": "RV280 [Radeon 9200 PRO / 9250]",
    "10025961": "RV280 [Radeon 9200]",
    "10025962": "RV280 [Radeon 9200]",
    "10025964": "RV280 [Radeon 9200 SE]",
    "10025940": "RV280 [Radeon 9200 PRO] (Secondary)",
    "10025941": "RV280 [Radeon 9200] (Secondary)",
    "10025944": "RV280 [Radeon 9200 SE PCI]",
    "10025954": "RS480 [Radeon Xpress 200 Series]",
    "10025955": "RS480M [Mobility Radeon Xpress 200]",
    "10025834": "RS300 [Radeon 9100 IGP]",
    "10025835": "RS300M [Mobility Radeon 9100 IGP]",
    "10025854": "RS480 [Radeon Xpress 200 Series] (Secondary)",
    "10025874": "RS480 [Radeon Xpress 1150] (Secondary)",
    "1002564B": "RV410/M26 GL [Mobility FireGL V5000]",
    "1002564F": "RV410/M26 [Mobility Radeon X700 XL]",
    "10025652": "RV410/M26 [Mobility Radeon X700]",
    "10025653": "RV410/M26 [Mobility Radeon X700]",
    "10025654": "Mach64 VT [Video Xpression]",
    "10025655": "264VT3 [Mach64 VT3]",
    "10025656": "Mach64 VT4 [Video Xpression+]",
    "10025657": "RV410 [Radeon X550 XTX / X700]",
    "1002554E": "R430 [All-In-Wonder X800 GT]",
    "1002554F": "R430 [Radeon X800]",
    "10025550": "R423 GL [FireGL V7100]",
    "10025551": "R423 GL [FireGL V5100]",
    "10025569": "R423 [Radeon X800 PRO] (Secondary)",
    "1002556B": "R423 [Radeon X800 GT] (Secondary)",
    "1002556D": "R480 [Radeon X800 GTO2/XL] (Secondary)",
    "1002556F": "R430 [Radeon X800] (Secondary)",
    "10025571": "R423 GL [FireGL V5100] (Secondary)",
    "10025462": "RV380/M24C [Mobility Radeon X600 SE]",
    "10025464": "RV370/M22 GL [Mobility FireGL V3100]",
    "10025549": "R423 [Radeon X800 GTO]",
    "1002554A": "R423 [Radeon X800 XT Platinum Edition]",
    "1002554B": "R423 [Radeon X800 GT/SE]",
    "1002554D": "R480 [Radeon X800 GTO2/XL]",
    "1002524C": "Rage 128 VR AGP",
    "1002534D": "Rage 128 4X AGP 4x",
    "10025354": "Mach 64 VT",
    "10025446": "Rage 128 PRO Ultra AGP 4x",
    "10025452": "Rage 128 PRO Ultra4XL VR-R AGP",
    "10025460": "RV370/M22 [Mobility Radeon X300]",
    "10025461": "RV370/M22 [Mobility Radeon X300]",
    "1002514C": "R200 [Radeon 8500/8500 LE]",
    "1002514D": "R200 [Radeon 9100]",
    "10025157": "RV200 [Radeon 7500/7500 LE]",
    "10025159": "RV100 [Radeon 7000 / Radeon VE]",
    "1002515E": "ES1000",
    "10025245": "Rage 128 GL PCI",
    "10025246": "Rage 128 (Rage 4) series",
    "1002524B": "Rage 128 VR PCI",
    "10024E6A": "RV350 [Radeon 9800 XT] (Secondary)",
    "10024E71": "RV350/M10 [Mobility Radeon 9600] (Secondary)",
    "10024F72": "RV250 [Radeon 9000 Series]",
    "10024F73": "RV250 [Radeon 9000 Series] (Secondary)",
    "10025044": "All-In-Wonder 128 PCI",
    "10025046": "Rage 4 [Rage 128 PRO AGP 4X]",
    "10025050": "Rage 4 [Rage 128 PRO PCI / Xpert 128 PCI]",
    "10025052": "Rage 4 [Rage 128 PRO AGP 4X]",
    "10025144": "R100 [Radeon 7200 / All-In-Wonder Radeon]",
    "10025148": "R200 GL [FireGL 8800]",
    "10024E54": "RV350/M10 GL [Mobility FireGL T2]",
    "10024E56": "RV360/M12 [Mobility Radeon 9550]",
    "10024E64": "R300 [Radeon 9700 PRO] (Secondary)",
    "10024E65": "R300 [Radeon 9500 PRO] (Secondary)",
    "10024E66": "RV350 [Radeon 9600] (Secondary)",
    "10024E67": "R300 GL [FireGL X1] (Secondary)",
    "10024E68": "R350 [Radeon 9800 PRO] (Secondary)",
    "10024E69": "R350 [Radeon 9800] (Secondary)",
    "10024E48": "R350 [Radeon 9800 Series]",
    "10024E49": "R350 [Radeon 9800]",
    "10024E4A": "R360 [Radeon 9800 XXL/XT]",
    "10024E4B": "R350 GL [FireGL X2 AGP Pro]",
    "10024E50": "RV350/M10 / RV360/M11 [Mobility Radeon 9600 (PRO) / 9700]",
    "10024E51": "RV350 [Radeon 9550/9600/X1050 Series]",
    "10024E52": "RV350/M10 [Mobility Radeon 9500/9700 SE]",
    "10024C66": "RV250/M9 GL [Mobility FireGL 9000/Radeon 9000]",
    "10024C6E": "RV250/M9 [Mobility Radeon 9000] (Secondary)",
    "10024D46": "Rage Mobility 128 AGP 4X/Mobility M4",
    "10024D52": "Theater 550 PRO PCI [ATI TV Wonder 550]",
    "10024D53": "Theater 550 PRO PCIe",
    "10024E44": "R300 [Radeon 9700/9700 PRO]",
    "10024E45": "R300 [Radeon 9500 PRO/9700]",
    "10024E46": "R300 [Radeon 9600 TX]",
    "10024E47": "R300 GL [FireGL X1]",
    "10024C4E": "Rage Mobility L AGP 2x",
    "10024C46": "Rage Mobility 128 AGP 2X/Mobility M3",
    "10024C47": "3D Rage IIC PCI / Mobility Radeon 7500/7500C",
    "10024C49": "3D Rage LT PRO PCI",
    "10024C4D": "Rage Mobility AGP 2x Series",
    "10024C50": "Rage 3 LT [3D Rage LT PRO PCI]",
    "10024C52": "M1 [Rage Mobility-M1 PCI]",
    "10024C54": "264LT [Mach64 LT]",
    "10024C57": "RV200/M7 [Mobility Radeon 7500]",
    "10024C58": "RV200/M7 GL [Mobility FireGL 7800]",
    "10024C59": "RV100/M6 [Rage/Radeon Mobility Series]",
    "10024A6A": "R420 [Radeon X800] (Secondary)",
    "10024A6B": "R420 [Radeon X800 XT AGP] (Secondary)",
    "10024A70": "R420 [Radeon X800 XT Platinum Edition AGP] (Secondary)",
    "10024A74": "R420 [Radeon X800 VE] (Secondary)",
    "10024B49": "R481 [Radeon X850 XT AGP]",
    "10024B4B": "R481 [Radeon X850 PRO AGP]",
    "10024B4C": "R481 [Radeon X850 XT Platinum Edition AGP]",
    "10024B69": "R481 [Radeon X850 XT AGP] (Secondary)",
    "10024B6B": "R481 [Radeon X850 PRO AGP] (Secondary)",
    "10024B6C": "R481 [Radeon X850 XT Platinum Edition AGP] (Secondary)",
    "10024C42": "Mach64 LT [3D Rage LT PRO AGP]",
    "1002496E": "RV250 [Radeon 9000] (Secondary)",
    "10024A49": "R420 [Radeon X800 PRO/GTO AGP]",
    "10024A4A": "R420 [Radeon X800 GT AGP]",
    "10024A4B": "R420 [Radeon X800 AGP Series]",
    "10024A4D": "R420 GL [FireGL X3-256]",
    "10024A4E": "RV420/M18 [Mobility Radeon 9800]",
    "10024A4F": "R420 [Radeon X850 AGP]",
    "10024A50": "R420 [Radeon X800 XT Platinum Edition AGP]",
    "10024A54": "R420 [Radeon X800 VE AGP]",
    "10024A69": "R420 [Radeon X800 PRO/GTO] (Secondary)",
    "10024966": "RV250 [Radeon 9000 Series]",
    "10024753": "Rage XC",
    "10024754": "Mach64 GT/GT-B [3D Rage I/II]",
    "10024755": "Mach64 GT-B [3D Rage II+ DVD]",
    "10024756": "Rage 2 [3D Rage IIC PCI]",
    "10024757": "Rage 2 [3D Rage IIC AGP]",
    "10024758": "Mach64 GX [WinTurbo]",
    "10024759": "Rage 3 [3D Rage IIC PCI]",
    "1002475A": "3D Rage IIC AGP",
    "10024654": "Mach64 VT",
    "10024742": "Rage 3 [3D Rage PRO AGP 2X]",
    "10024744": "Rage 3 [3D Rage PRO AGP 1X]",
    "10024749": "3D Rage PRO PCI",
    "1002474D": "Rage XL AGP 2X",
    "1002474E": "Rage XC AGP",
    "1002474F": "Rage XL",
    "10024750": "3D Rage Pro PCI",
    "10024752": "Rage 3 [Rage XL PCI]",
    "10024437": "RS250 [Mobility Radeon 7000 IGP]",
    "10024554": "210888ET [Mach64 ET]",
    "10024630": "XENOS Parent Die (XBOX 360)",
    "10024631": "XENOS Daughter Die (XBOX 360)",
}


update_modules()

if len(sys.argv) > 1 and sys.argv[1]=="--launch-uvicorn":
    launch_uvicorn()
