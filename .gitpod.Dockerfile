FROM gitpod/workspace-full:2022-07-09-12-00-00

# Install custom tools, runtime, etc.
RUN sudo install-packages gdal-bin libgdal-dev 