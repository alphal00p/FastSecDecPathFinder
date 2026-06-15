#!/bin/sh
set -eu

pid_file="${1:-/tmp/fsd_kill_pids}"

if [ ! -s "$pid_file" ]; then
  exit 0
fi

xargs kill < "$pid_file"
