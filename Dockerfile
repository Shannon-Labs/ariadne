FROM --platform=linux/arm64/v8 mambaorg/micromamba:1.5.8

USER root
WORKDIR /workspace

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

COPY . /workspace

RUN python -m pip install -e .

CMD ["bash"]

