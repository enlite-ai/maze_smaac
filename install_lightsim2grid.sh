#!/bin/bash

git clone https://github.com/BDonnot/lightsim2grid.git
(
  cd lightsim2grid || exit
  git checkout v0.4.0
  git submodule init
  git submodule update
  make
  pip install -U pybind11
  pip install -U .
)