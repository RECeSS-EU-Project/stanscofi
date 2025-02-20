Installation of *stanscofi*
----------------------------

The complete list of dependencies for *stanscofi* can be found at `requirements.txt <https://raw.githubusercontent.com/RECeSS-EU-Project/stanscofi/master/pip/requirements.txt>`_ (Pip) or `meta.yaml <https://raw.githubusercontent.com/RECeSS-EU-Project/stanscofi/master/conda/meta.yaml>`_ (Conda). This package is compatible with Python 3.8 and 3.9, and should at least run on Linux.

Running the Docker image
:::::::::::::::::::::::::

You can download the Docker image hosted on DockerHub and start a container on this image: ::

    $ docker push recessproject/stanscofi:<release-version>
    $ docker run -it recessproject/stanscofi:<release-version>
    
To build the image locally and run the notebooks, please use the following commands at the root folder (Credits to `Abhishek Tiwari <https://github.com/abhishektiwari/>`_ for the Dockerfile, instructions and comments): ::

    $ #Build Docker image
    $ docker build -t stanscofi . 
    $ #Run Docker image built in previous step and drop into SSH
    $ docker run -it --expose 3000  -p 3000:3000 stanscofi 
    $ #Run notebook
    $ #The notebook is available at http://127.0.0.1:3000/tree.
    $ cd docs && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 3000 

Python package managers (Pip or Conda)
::::::::::::::::::::::::::::::::::::::::

It is recommended to use a virtual environment (here, we give an example using Conda): ::

    $ conda create -n stanscofi_env python=3.8.5 -y
    $ conda activate stanscofi_env
    $ python3 -m pip install stanscofi ## if using pip
    $ conda install -c recess stanscofi ## if using conda
    $ ## useful if you want to run the Jupyter notebooks in the docs/ folder
    $ python3 -m pip install notebook>=6.5.4 markupsafe==2.0.1 
    $ conda deactivate && conda activate stanscofi_env

Manual installation (not recommended)
::::::::::::::::::::::::::::::::::::::

Download the `tar.gz file from PyPI <https://pypi.python.org/pypi/stanscofi/>`_ and extract it. It is recommended to use a virtual environment. The library consists of a directory named `stanscofi` containing several Python modules. Then execute the following commands at the root folder: ::

    $ python3 -m pip install --upgrade pip wheel
    $ python3 -m pip install -r pip/requirements.txt
    $ python3 -m pip install .
