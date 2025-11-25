FROM tercen/runtime-python39:0.1.0

COPY . /operator
WORKDIR /operator

ENV PYTHONPATH "${PYTHONPATH}:~/.pyenv/versions/3.9.0/bin/python3"

# Install the package
RUN python3 -m pip install .

# Set the entrypoint to the CLI command
ENTRYPOINT ["model-estimator"]