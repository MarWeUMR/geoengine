FROM gitpod/workspace-full:latest

# Install custom tools, runtime, etc.
RUN sudo install-packages gdal-bin libgdal-dev
RUN curl -LO https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.deb
RUN sudo apt install ./nvim-linux64.deb
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y
RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim\
 ~/.local/share/nvim/site/pack/packer/start/packer.nvim
WORKDIR /home/gitpod

RUN git clone https://github.com/MarWeUMR/nvim

