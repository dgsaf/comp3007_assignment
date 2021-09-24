#!/bin/bash

# Improve script safety.
set -o errexit
set -o nounset
set -o pipefail

# Script messaging.
FLAG_MSG_DEBUG="1"
FLAG_MSG_ERROR="1"

function __msg_debug() {
  if [ "${FLAG_MSG_DEBUG}" == "1" ] ; then
    echo -e "[DEBUG] $*"
  fi
}

function __msg_error() {
  if [ "${FLAG_MSG_ERROR}" == "1" ] ; then
    echo -e "[ERROR] $*"
  fi
}
