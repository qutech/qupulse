import sys
import subprocess

__all__ = ["install_requirements"]

try:
    from qupulse.hardware.awgs_new_driver.tabor import TaborDevice, TaborChannelTuple
    __all__.extend(["TaborAWGRepresentation", "TaborChannelPair"])
except ImportError:
    pass

try:
    from qupulse.hardware.awgs.tektronix import TektronixAWG
    __all__.extend(["TektronixAWG"])
except ImportError:
    pass


def install_requirements(vendor: str):
    package_repos = {
        'tektronix': 'tek_awg',
        'tabor': 'https://git.rwth-aachen.de/qutech/python-TaborDriver/-/archive/python3/python-TaborDriver-python3.zip'
    }

    if vendor not in package_repos:
        raise ValueError('Vendor must be in {}'.format(set(package_repos.keys())))

    repo = package_repos[vendor]
    subprocess.check_call([sys.executable, "-m", "pip", "install", repo])
