# Build image with multi-stage builds
# Ensures image is built faster and smaller
FROM python:3.8-slim AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

# Install with pip‘s --user option, so all files will be installed in
# the .local directory of the current user’s home directory
COPY . /ColonyScanalyser
WORKDIR /ColonyScanalyser
RUN pip install --user .

# Copy compiled artefacts to fresh image, without compiler baggage
FROM python:3.8-slim AS build-image
COPY --from=compile-image /root/.local /root/.local

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Return help if no arguments are passed to ColonyScanalyser
ENTRYPOINT ["colonyscanalyser"]
CMD ["-h"]