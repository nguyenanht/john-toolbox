#!/bin/bash

URL_NOTEBOOK="https://nb.johntoolbox.localhost/"

echo $URL_NOTEBOOK

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  xdg-open $URL_NOTEBOOK
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # Mac OSX
  open $URL_NOTEBOOK
elif [[ "$OSTYPE" == "cygwin" ]]; then
    # POSIX compatibility layer and Linux environment emulation for Windows
  start $URL_NOTEBOOK
elif [[ "$OSTYPE" == "msys" ]]; then
    # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
  start $URL_NOTEBOOK
else
  echo "$OSTYPE" not handled.
fi;