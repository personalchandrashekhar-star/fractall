"""Files that the three accepted models fetch from the internet when they load, served from the lab instead.

ACT builds its image encoder from torchvision's ResNet-18 and asks for that file (45 MB) whenever a policy is
loaded, also a trained one. SmolVLA builds on the SmolVLM2 model and asks the Hugging Face Hub for it (1.9 GB).
Diffusion Policy fetches nothing. The lab holds both files (``LAB_MODELS``). ``link_lab_models`` puts them where
PyTorch and the Hugging Face libraries look, as links, unless a copy is there already; ``go_offline_if_needed``
tells the Hugging Face libraries not to wait for a server that cannot be reached. With both, a policy loads without
the internet. Where the lab holds no such files, nothing changes: the libraries download what they need.

    python -m nwaves_hackathon.models          # link the files and say what was done (nwaves-train does this)
"""
import os
import socket

LAB_MODELS = os.environ.get("NWAVES_MODELS") or "/opt/nwaves/models"


def _workspace() -> str:
    return os.environ.get("WORKSPACE") or os.path.expanduser("~/workspace")


def torch_home() -> str:
    return os.environ.get("TORCH_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "torch")


def hf_hub_cache() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def link_lab_models() -> list:
    """Link every model file the lab holds into the caches of PyTorch and of the Hugging Face libraries, unless
    the cache has it already. Returns the names that were linked. Never raises: a cache that cannot be written is
    left alone."""
    linked = []
    for source, target in ((os.path.join(LAB_MODELS, "torch", "hub", "checkpoints"), os.path.join(torch_home(), "hub", "checkpoints")),
                           (os.path.join(LAB_MODELS, "hf", "hub"), hf_hub_cache())):
        if not os.path.isdir(source):
            continue
        try:
            os.makedirs(target, exist_ok=True)
            for name in sorted(os.listdir(source)):
                if not os.path.lexists(os.path.join(target, name)):
                    os.symlink(os.path.join(source, name), os.path.join(target, name))
                    linked.append(name)
        except OSError:
            pass
    return linked


def online(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def go_offline_if_needed() -> bool:
    """Without a connection to the Hugging Face Hub, tell its libraries to use what is on disk and not to wait.
    Call before anything imports them. Returns True when the lab is offline."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return True
    if online():
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return True


if __name__ == "__main__":
    names = link_lab_models()
    if names:
        print(f"[nwaves] model files of this lab linked into your caches: {', '.join(names)}", flush=True)
