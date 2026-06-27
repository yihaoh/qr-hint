FROM postgres:16
ARG USER
ARG GROUP
ARG PASSWD

RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

RUN groupadd --gid 1001 ${GROUP}
RUN useradd --home /home/${USER} --shell /bin/bash --uid 1001 --gid 1001 ${USER}
RUN echo "${USER}:${PASSWD}" | chpasswd
RUN apt-get update
RUN apt-get -y install sudo 
RUN usermod -aG sudo ${USER}
RUN echo "${USER} ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee "/etc/sudoers.d/dont-prompt-$USER-for-sudo-password"

COPY dotfiles/* /home/${USER}/
RUN chown -R ${USER}:${GROUP} /home/${USER}

COPY install.sh /opt/install.sh
RUN /bin/bash -c /opt/install.sh
COPY install-as-user.sh /opt/install-as-user.sh
RUN sudo -u ${USER} /bin/bash -c /opt/install-as-user.sh

# cannot uncomment the following; postgres image seems to have post-processing that assume root:
# USER ${USER}
# WORKDIR /home/${USER}