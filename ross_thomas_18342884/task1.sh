#!/bin/bash

# Initialise top directory.
DIR_TOP="/home/student/Documents/COMP3007/comp3007_assignment" # "/home/student"

# Initialise directory variables.
DIR_SUBMISSION="${DIR_TOP}/ross_thomas_18342884"
DIR_DIGITS="${DIR_TOP}/train/digits" # "${DIR_TOP}/digits"
DIR_TRAIN="${DIR_TOP}/train/task1"
DIR_TEST="${DIR_TOP}/test/task1"
DIR_VAL="${DIR_TOP}/val/task1"
DIR_OUTPUT="${DIR_SUBMISSION}/output/task1"
DIR_WORK="${DIR_SUBMISSION}/work"

# Source common scripting functionality.
source "${DIR_SUBMISSION}/common.sh"

# Log directory variables.
__msg_debug "Directories:"
# for name in TOP SUBMISSION DIGITS TEST OUTPUT ; do
for name in TOP SUBMISSION DIGITS TRAIN TEST VAL OUTPUT WORK ; do
  dir="DIR_${name}"

  if [ -d "${!dir}" ] ; then
    __msg_debug "> ${dir}=${!dir}"
  else
    __msg_error "> ${dir}=${!dir} does not exist"
    exit 1
  fi
done

# Run `task_1.py` script.
# Swap DIR_TRAIN with DIR_TEST for final run.
/usr/bin/python3 \
  "${DIR_SUBMISSION}/src/task_1.py" \
  -i "${DIR_TRAIN}" \
  -o "${DIR_OUTPUT}" \
  -d "${DIR_DIGITS}" \
  -w "${DIR_WORK}" -W
