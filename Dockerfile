FROM yandex/rep:0.6.6
MAINTAINER Alex Rogozhnikov <axelr@yandex-team.ru>

RUN sudo apt-get update
RUN sudo apt-get install -y graphviz
RUN /root/miniconda/envs/rep_py2/bin/pip install pydot-ng
RUN /root/miniconda/envs/rep_py2/bin/pip install keras==1.2.0
