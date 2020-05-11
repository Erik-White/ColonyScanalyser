# Run a containerised instance of ColonyScanalyser on
# a directory of images in the host system
#
# Pass the directory path to this script, followed by any
# other command line arguments for ColonyScanalyser
# The directory path must be absolute, not relative
#
# Example:
# docker-run.sh /path/to/images/ --verbose 5 --plate_size 100

# Store images directory and mount in docker image
# Shift remaining arguments and pass to ColonyScanalyser
img_dir="$1"
shift
# Run ColonyScanalyser on the images in the mounted directory
docker run \
  --rm \
  --name colonyscanalyser \
  --mount type=bind,source=$img_dir,target=/data \
  colonyscanalyser /data \
  $@