import sys
import subprocess

__all__ = ["install_requirements"]

try:
    from qupulse.hardware.awgs.tabor import TaborAWGRepresentation, TaborChannelPair
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
        'tabor': 'tabor_control'
    }

    if vendor not in package_repos:
        raise ValueError('Vendor must be in {}'.format(set(package_repos.keys())))

    repo = package_repos[vendor]
    subprocess.check_call([sys.executable, "-m", "pip", "install", repo])
