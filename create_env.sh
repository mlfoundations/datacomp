#!/bin/bash

ARCH=`uname -m`
if [ "$ARCH" = "x86_64" ]; then
    echo "Intel architecture detected."
    conda env create -f environment.yml
elif [ "$ARCH" = "arm64" ]; then
    echo "Apple M1 architecture detected."
    conda env create -f environment_osx.yml
else
    echo "Unknown architecture: $ARCH"
fi
