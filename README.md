# DL-Idempotent
For ideal usability please use this environment when working with the code

### Environment Setup

1. Open a terminal and navigate to the project directory.
2. Use the following command to create the Conda environment from the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
3. Run the following command to strip outputs from jupyter notebooks to avoid issues with git:
   ```bash
   nbstripout --install
### Update your Environment

1. Open a terminal and navigate to the project directory.
2. Run he following command
   ```bash
   conda env update -f environment.yml
### Publish changes to the Environment
Whenever you install a new package to the environment follow these steps to guarantee that the environment is always at the newest status
1. Open a terminal and navigate to the project directory.
2. Export your conda environment
   ```bash
   conda env export > environment.yml
3. Open environment.yml in a text editor and delete the last row `prefix`
4. Push the updated `environment.yml`

