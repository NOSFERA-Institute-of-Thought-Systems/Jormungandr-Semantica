# Use a standard, slim base image
FROM ubuntu:22.04

# Set frontend to noninteractive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities like wget and set up the shell
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (a minimal conda installer)
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Make conda available in the PATH
ENV PATH /opt/conda/bin:$PATH

# Copy our environment definition into the image
COPY environment.yml .

# Create the conda environment from the file
RUN conda env create -f environment.yml

# Activate the conda environment for subsequent commands
SHELL ["conda", "run", "-n", "jormungandr", "/bin/bash", "-c"]

# Create a directory for our project code
WORKDIR /app
COPY . .

# Build and install our project inside the environment
# This wheel will be a Linux (manylinux) wheel
RUN pip install . --no-cache-dir

# Set a default command to run when the container starts
CMD ["python", "scripts/verify_install.py"]