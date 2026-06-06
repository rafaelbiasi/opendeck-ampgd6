![Plugin Icon](assets/icon.png)

# OpenDeck FIFINE Ampligame D6 Plugin

An unofficial plugin for FIFINE Ampligame D6 device

## OpenDeck version

Requires OpenDeck 2.5.0 or newer

## Supported devices

- FIFINE Ampligame D6 (3142:0007)
- FIFINE Ampligame D6 Rev. 2 (3142:0060)

## Known issues

- The D6 currently behaves like a protocol v1 device for identity purposes, so using two identical D6 units at the same time is not supported.
- If you used a pre-`0.2.0` build of this plugin, OpenDeck may not reuse old page bindings automatically because the device namespace changed from `99` to `d6`.

## Platform support

- Linux: Guaranteed, if stuff breaks - I'll probably catch it before public release
- Mac: Best effort, no tests before release, things may break, but I probably have means to fix them
- Windows: Zero effort, no tests before release, if stuff breaks - too bad, it's up to you to contribute fixes

## Installation

1. Download an archive from [releases](https://github.com/shugotekitten/opendeck-ampgd6/releases)
2. In OpenDeck: Plugins -> Install from file
3. Linux: Download [udev rules](./40-opendeck-ampgd6.rules) and install them by copying into `/etc/udev/rules.d/` and running `sudo udevadm control --reload-rules`
4. Unplug and plug again the device, restart OpenDeck

## Device specifications

- Layout: 3 rows × 5 columns (15 buttons)
- Protocol version: 1
- Protocol version: 3 (D6 Rev. 2)

## Building

### Prerequisites

You'll need:

- A Linux OS of some sort
- Rust 1.87 and up with `x86_64-unknown-linux-gnu` and `x86_64-pc-windows-gnu` targets installed
- Docker
- [just](https://just.systems)

### Preparing environment

```sh
$ just prepare
```

This will build docker image for macOS crosscompilation

### Building a release package

```sh
$ just package
```

## Acknowledgments

This plugin is heavily based on work by contributors of [elgato-streamdeck](https://github.com/streamduck-org/elgato-streamdeck) crate
