#!/bin/bash

# Source common scripting functionality.
source "./common.sh"

# Initialise top directory.
DIR_TOP="/home/student"
DIR_TOP="$(dirname "${PWD}")"

# Initialise directory variables.
DIR_SUBMISSION="${DIR_TOP}/ross_thomas_18342884"
DIR_TRAIN="${DIR_TOP}/train/task1"
DIR_TEST="${DIR_TOP}/test/task1"
DIR_VAL="${DIR_TOP}/val/task1"

# Log directory variables.
__msg_debug "Directories:"
for name in TOP SUBMISSION TRAIN TEST VAL ; do
  dir="DIR_${name}"

  if [ -d "${!dir}" ] ; then
    __msg_debug "> ${dir}=${!dir}"
  else
    __msg_error "> ${dir}=${!dir} does not exist"
    exit 1
  fi
done
