#!/bin/bash

# Add PPA
sudo add-apt-repository ppa:jonathonf/python-3.6

# Update all installed packages
sudo apt-get update -qq

# Install python 3.6
sudo apt-get -y install python3.6

# Install and upgrade pip
sudo apt-get -y install python3-pip
python3.6 -m pip install --upgrade pip

# Install tkinter
sudo apt-get -y install python3.6-tk

# Install requirements
pip install --user -r /vagrant/requirements.txt
