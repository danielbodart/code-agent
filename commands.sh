#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Just in time installers, these functions replace themselves with the real command when called for the first time

function curl() {
  unset -f curl
  if [[ ! $(command -v curl) ]]; then
    sudo apt install -y curl
  fi
  curl "$@"
}
export -f curl



function uv() {
  unset -f uv
  if [[ ! -f "$HOME/.local/bin/uv" ]]; then
    echo "* Downloading and installing mise..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  uv "$@"
}
export -f uv


function python() {
  uv run --script "$@"
 }

# If this script is being executed (not sourced) and has an argument, run it with bun
if [[ "${BASH_SOURCE[0]}" == "${0}" && -n "${1}" ]]; then
    python "$@"
    exit $?
fi