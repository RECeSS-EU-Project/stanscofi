# Pull base image.
FROM python:3.8.18-bullseye

# Install.
RUN \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y byobu curl git htop man unzip vim wget pip && \
  pip install stanscofi notebook

# Add files.
ADD . /home/stanscofi

# Set environment variables.
ENV HOME /home

# Define working directory.
WORKDIR /home/stanscofi

# Define default command.
CMD ["bash"]

EXPOSE 3000