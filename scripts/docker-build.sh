# Create a containerised image of ColonyScanalyser
docker build --pull --rm -f "./Dockerfile" -t colonyscanalyser:latest "."