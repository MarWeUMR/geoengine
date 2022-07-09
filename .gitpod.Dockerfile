FROM gitpod/workspace-full:2022-07-09-12-00-00

# Install custom tools, runtime, etc.
RUN sudo install-packages gdal-bin libgdal-dev 
RUN curl -LO https://github.com/neovim/neovim/releases/download/nightly/nvim.appimage
RUN chmod u+x nvim.appimage
RUN ./nvim.appimage --appimage-extract
RUN ./squashfs-root/AppRun --version

RUN ln -s /squashfs-root/AppRun /usr/bin/nvim
RUN rm nvim.appimage