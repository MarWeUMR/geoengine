FROM gitpod/workspace-full:latest

# Install custom tools, runtime, etc.
RUN sudo install-packages gdal-bin libgdal-dev 
RUN curl -LO https://github.com/neovim/neovim/releases/download/nightly/nvim.appimage
RUN chmod u+x nvim.appimage
RUN ./nvim.appimage --appimage-extract
RUN ./squashfs-root/AppRun --version

RUN sudo ln -s /squashfs-root/AppRun /usr/bin/nvim
RUN rm nvim.appimage