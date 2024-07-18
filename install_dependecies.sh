#!/bin/bash

# Install dependencies for the project
sudo apt install libjpeg-dev zlib1g-dev

# Install requirements
pip install -r requirements.txt

# Check if git submodule was recursively cloned
git submodule update --init --recursive

# Navigate to RepViT submodule and install package
cd RepViT
pip install -e .

echo "Dependencies installed successfully"


