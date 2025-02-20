Installation of *benchscofi*
----------------------------

The complete list of dependencies for *benchscofi* can be found at `requirements.txt <https://raw.githubusercontent.com/RECeSS-EU-Project/benchscofi/master/pip/requirements.txt>`_ (Pip). This package is compatible with Python 3.8, and should at least run on Linux.

Running the Docker image (easiest)
:::::::::::::::::::::::::::::::::::

You can download the Docker image hosted on DockerHub and start a container on this image: ::

    $ docker push recessproject/benchscofi:<release-version>
    $ docker run -it recessproject/benchscofi:<release-version>
    
Manual installation (not recommended)
::::::::::::::::::::::::::::::::::::::

It is recommended to use a virtual environment (here, we give an example using Conda): ::

    $ conda create -n benchscofi_env python=3.8.5 -y
    $ conda activate benchscofi_env
    $ # Using Pip
    $ python3 -m pip install benchscofi
    $ python3 -m pip uninstall werkzeug
    $ ## useful if you want to run the Jupyter notebooks in the docs/ folder
    $ python3 -m pip install notebook>=6.5.4 markupsafe==2.0.1
    $ conda deactivate && conda activate benchscofi_env
    
Then install the dependencies

- Install R based on your distribution, or do not use the following algorithms: ``LRSSL``. Check if R is properly installed using the following command: ::

   $ R -q -e "print('R is installed and running.')

- Install MATLAB or Octave (free, with packages ``statistics`` from Octave Forge) based on your distribution, or do not use the following algorithms: ``BNNR`` (depends on Octave), ``SCPMF`` (depends on Octave), ``MBiRW`` (depends on Octave), ``DDA_SKF`` (depends on Octave and package ``statistics``). Check if Octave is properly installed using the following command ::

  $ octave --eval "'octave is installed'"
  $ octave --eval "pkg load statistics; 'octave-statistics is installed'"

- Install a MATLAB compiler (version 2012b) as follows, or do not use algorithm ``DRRS``:  ::

  $ apt-get install -y libxmu-dev libncurses5 # libXmu.so.6 and libncurses5 are required
  $ wget -O MCR_R2012b_glnxa64_installer.zip \
  $ https://ssd.mathworks.com/supportfiles/MCR_Runtime/R2012b/MCR_R2012b_glnxa64_installer.zip
  $ mv MCR_R2012b_glnxa64_installer.zip /tmp
  $ cd /tmp
  $ unzip MCR_R2012b_glnxa64_installer.zip -d MCRInstaller
  $ cd MCRInstaller
  $ mkdir -p /usr/local/MATLAB/MATLAB_Compiler_Runtime/v80
  $ chown -R <user> /usr/local/MATLAB/
  $ ./install -mode silent -agreeToLicense  yes

Installation notes 
::::::::::::::::::::

--
    
Instead of using Pip (command "python3 -m pip install benchscofi"), one might need to download from source files. In that case, download the `tar.gz file from PyPI <https://pypi.python.org/pypi/benchscofi/>`_ and extract it (or clone the GitHub repository, for the latest non-stable release): ::

    $ git clone https://github.com/RECeSS-EU-Project/benchscofi.git
    $ cd benchscofi/
    $ python3 -m pip install --upgrade pip wheel
    $ python3 -m pip install -r pip/requirements.txt
    $ python3 -m pip install .
    
--
    
An `issue <https://github.com/RECeSS-EU-Project/benchscofi/issues/1>`_ might arise with the version of Tensorflow provided in the *requirements.txt*. In fact, the requirement for *tensorflow* might be too restrictive. In that case, rely on the dependency conflict solver of *pip* (which might take a while, but will successfully solve everything) and proceed as follows:

- Replace the *install_requires* field in file *setup.py* by ::

    install_requires=["stanscofi", "tensorflow", "pulearn",
     "torch", "fastai", "torch_geometric", "pyFFM", "pytorch-lightning", "scikit-learn==1.2.*",
     "libmf"]

and then run the following commands: ::

    $ git clone https://github.com/RECeSS-EU-Project/benchscofi.git
    $ cd benchscofi/ && python3 -m pip install .
    
--
