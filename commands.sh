#!/usr/bin/env bash


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

function pip() {
  uv pip "$@"
 }

function venv() {
  uv venv "$@"
}

# If this script is being executed (not sourced) and has an argument, run it
if [[ "${BASH_SOURCE[0]}" == "${0}" && -n "${1}" ]]; then
    python "$@"
    exit $?
fi