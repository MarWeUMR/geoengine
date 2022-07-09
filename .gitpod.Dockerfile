FROM gitpod/workspace-full:latest

# Install custom tools, runtime, etc.
RUN sudo install-packages gdal-bin libgdal-dev 
RUN curl -LO https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.deb
RUN sudo apt install ./nvim-linux64.deb
