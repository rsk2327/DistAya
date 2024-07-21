import torch

def getmodule(module: torch.nn.Module, target_module: str):
    """Get a target module from a given module."""
    submodules = target_module.split(".", 1)
    if submodules[0].isdigit():
      next_module = module[int(submodules[0])]
    else:
      next_module = getattr(module, submodules[0])
    if len(submodules) == 1:
        return next_module
    return getmodule(next_module, submodules[-1])

def setmodule(module: torch.nn.Module, target_module: str, value: torch.nn.Module):
    """Set a target module in a given module."""
    submodules = target_module.split(".", 1)
    if len(submodules) == 1:
        if submodules[0].isdigit():
            module[int(submodules[0])] = value
        else:
            setattr(module, submodules[0], value)
    else:
        setmodule(getattr(module, submodules[0]), submodules[-1], value)