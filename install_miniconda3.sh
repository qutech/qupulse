#!/bin/sh

set -e
if [ ! -d "$HOME/Downloads" ]; then
  mkdir $HOME/Downloads
fi

if [ ! -e "$HOME/Downloads/miniconda3.sh" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/Downloads/miniconda3.sh
  chmod +x $HOME/Downloads/miniconda3.sh
else
  echo 'Using cached miniconda installer.';
fi

$HOME/Downloads/miniconda3.sh -b

rm -r -f $HOME/miniconda3/pkgs
ln -s $HOME/.cache/miniconda3_pkgs $HOME/miniconda3/pkgs


