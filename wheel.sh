#! /bin/bash

if [ -d dist ]; then
    rm -r dist
fi
pip wheel -w dist .
cd dist
unzip *
cd ..
