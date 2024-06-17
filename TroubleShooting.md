## Trouble Shooting

Here, we document some potential errors we have encountered. Please review the following information before opening an issue.

### ManiSkill2 Installation

Generally, [ManiSkill's official trouble shooting guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting) is excellent. Here, we outline some regular errors on installation.

You can quickly test whether your ManiSkill2 is successfully installed by running `python -m mani_skill2.examples.demo_random_action`.

Specifically, if you are using a headless server, you should run `export DISPLAY=''`.

If you meet with one of the following errors:
- `RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed`
- `Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.`
- `Segmentation fault (core dumped)`

You can first check your NVIDIA drivers by running:
```bash
ldconfig -p | grep libGLX_nvidia
```
If `libGLX_nvidia.so` is not found, they it is likely that you have installed an incorrect driver. To get the right driver on linux, it is recommended to install `nvidia-driver-xxx` (do not use the ones with server in the package name) and to avoid using any other method of installation like a runfile.

If your NVIDIA driver is correct, then please test your Vulkan installation:
```bash
sudo apt-get install vulkan-utils
vulkaninfo | head -n 10
```

If `vulkaninfo` fails to show the information about Vulkan, please check whether the following files exist:
- `/usr/share/vulkan/icd.d/nvidia_icd.json`
- `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
- `/etc/vulkan/implicit_layer.d/nvidia_layers.json` (optional, but necessary for some GPUs like A100)

If `/usr/share/vulkan/icd.d/nvidia_icd.json` does not exist, try to create the file with the following content:
```
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
```

If `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` does not exist, you can try `sudo apt-get install libglvnd-dev`. `10_nvidia.json` should contain the following content:
```
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
```

If `/etc/vulkan/implicit_layer.d/nvidia_layers.json` does not exist, try to create the file with the following content:
```
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.277",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
```

More discussions can be found [here](https://github.com/haosulab/SAPIEN/issues/115).
