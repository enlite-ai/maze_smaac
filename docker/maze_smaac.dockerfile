# syntax=docker/dockerfile:1.2
ARG MAZE_CORE_ENV=maze_core_env:latest

FROM ${MAZE_CORE_ENV}

# Update conda environment.
WORKDIR /usr/src/maze_smaac
#RUN echo ${BASE_PATH}environment.yml
COPY environment.yml environment.yml
RUN mamba env update --name env -f environment.yml

# Install lightsim2grid.
WORKDIR /usr/src/
COPY install_lightsim2grid.sh .
RUN ./install_lightsim2grid.sh

# Copy code.
COPY . .

