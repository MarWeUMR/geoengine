# You can find the new timestamped tags here: https://hub.docker.com/r/gitpod/workspace-base/tags
FROM gitpod/workspace-rust:latest

# Install custom tools, runtime, etc.
# base image only got `apt` as the package manager
# install-packages is a wrapper for `apt` that helps skip a few commands in the docker env.

RUN sudo install-packages clang gdal-bin libgdal-dev lld cmake sqlite3 
RUN rustup toolchain install nightly 
RUN rustup default nightly
RUN sudo install-packages fzf ranger npm fish

# USE TO HAVE NVIM AVAILABLE DURING IMAGE BUILDING PROCESS
RUN curl -LO https://github.com/neovim/neovim/releases/download/nightly/nvim.appimage
RUN chmod u+x nvim.appimage
RUN ./nvim.appimage --appimage-extract
RUN ./squashfs-root/AppRun --version

RUN sudo ln -s /squashfs-root/AppRun /usr/bin/nvim

RUN npm install tree-sitter-cli
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y

WORKDIR /home/gitpod

RUN mkdir -p /home/gitpod/.config 
WORKDIR /home/gitpod/.config
RUN git clone https://github.com/MarWeUMR/nvim 
RUN git clone https://github.com/MarWeUMR/fish 
RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim\
  /home/gitpod/.local/share/nvim/site/pack/packer/start/packer.nvim

cmd ["/usr/bin/fish", "curl -sS https://git.io/fisher | source && fisher install jorgebucaran/fisher"] 