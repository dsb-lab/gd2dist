#! /bin/bash

pip wheel -w dist .
cd dist
unzip *
cd ..
python proba.py