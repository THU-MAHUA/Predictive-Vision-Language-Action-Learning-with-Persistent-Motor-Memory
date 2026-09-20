"""IPME-VLA public package."""

from ipme_vla.config import IPMEVLAConfig

__all__ = ["IPMEVLAConfig", "IPMEVLAPolicy"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "IPMEVLAPolicy":
        from ipme_vla.models.vla.policy import IPMEVLAPolicy

        return IPMEVLAPolicy
    raise AttributeError(name)
