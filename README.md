# DDPM-noise-types
Applying simple DDPM onto different noise types

Execution steps:
1. Create either a .venv or conda environment. (It is possible to run on Kaggle if combined into one notebook or sourced from the files appropriately)
2. Pip install requirements.txt (some pips were not added to the requirements and need to be added depending on the code you are to execute)
3. For different noise types, you need to execute 'python Diffusion_{noise_type}.py'


Problems:
1. Kernel dies - Solution: decrease the amount of data (like to 20000 images instead or less), change number of epochs.
