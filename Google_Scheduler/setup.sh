#!/bin/bash 

python -m pip install --upgrade --user ortools

#Then verify that you have pip 9.01 or higher available in your PATH:

python --version
python -c "import platform; print(platform.architecture()[0])"
python -m pip --version

# Actual installation( For windows and linux both but if you prefer system wide installation, follow the step on 15)
python -m pip install --user ortools

# Or if you want to install it systemwide on lInux follow the below command
# sudo python -m pip install -U ortools