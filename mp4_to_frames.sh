#!/bin/bash

find . -name "*.mp4" -exec sh -c 'mkdir -p "${1%.*}/frames" ; mv "$1" "${1%.*}"; ffmpeg -i "${1%.*}/$1" -vf fps=4 "${1%.*}/frames/${1%.*}_%04d.png" ' _ {} \;
