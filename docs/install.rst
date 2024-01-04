Installation
------------

With Conda
::::::::::

You can install the stanscofi package using pip: ::

    $ conda install -c recess stanscofi

With Pip
::::::::

You can install the stanscofi package using pip: ::

    $ pip install stanscofi
    
Using Docker image
::::::::::::::::::::::::

You can also use the Docker image hosted on DockerHub: ::

    $ docker push recessproject/stanscofi:<version>

From Source Files
:::::::::::::::::

Download the `tar.gz file from PyPI <https://pypi.python.org/pypi/stanscofi/>`_ and extract it.  The library consists of a directory named `stanscofi` containing several Python modules.

Using Docker
:::::::::::::

Credits to `Abhishek Tiwari <https://github.com/abhishektiwari/>`_ for the Dockerfile, instructions and comments. In the root folder of the repository, run the following commands: ::

    $ #Build Docker image
    $ docker build -t stanscofi . 
    $ #Run Docker image built in previous step and drop into SSH
    $ docker run -it --expose 3000  -p 3000:3000 stanscofi 
    $ #Run notebook
    $ cd docs && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 3000 

The notebook is available at http://127.0.0.1:3000/tree.
