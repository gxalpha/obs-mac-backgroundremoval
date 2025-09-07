# macOS Background Removal for OBS

## Introduction

This plugin provides an OBS filter to remove a persons background using macOS' built-in [Vision](https://developer.apple.com/documentation/vision) API.

This plugin requires OBS 31.1 or later.

## Getting started

To get started, download and run the installer from the [releases](https://github.com/gxalpha/obs-mac-backgroundremoval/releases) page.
It's currently not signed or notarized.
Because of that, follow these installation instructions:
- Open the `.pkg` file that you have downloaded. You will get prompted with a warning, select "Done" (do not move to trash).
- Open System Settings. Navigate to "Privacy & Security" → "Security". Near the bottom, you will find an information that `obs-mac-backgroundremoval-[...].pkg` was blocked.
- Click "Open Anyway" and confirm using your administrator login.

The filter can be found in the filters window under "Effect Filters".
It can be used on any type of source, not just Video Capture Devices!

## License and Thanks
This plugin is licensed under the terms of the General Public License, Version 2. You can find the full text in the `LICENSE` file in this repository.

Huge credits also go to pkv from OBS who implemented a similar filter in OBS for NVIDIA GPUs. His code was referenced heavily in the creation of this plugin, especially during the early stages.

The build system is based on the [OBS Plugin Template](https://github.com/obsproject/obs-plugintemplate). Its build instructions also apply to this plugin.
