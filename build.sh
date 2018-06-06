set -e
# I put my secrets is untracked bash file
source .secrets.env
docker login $SONIC_DOCKER_REGISTRY \
    --username $SONIC_DOCKER_USERNAME \
    --password $SONIC_DOCKER_PASSWORD

docker pull openai/retro-env
docker tag openai/retro-env remote-env

# Change these
SONIC_CONTAINER_NAME=wassname_curious
n=6

docker build -t $SONIC_DOCKER_REGISTRY/$SONIC_CONTAINER_NAME:v${n} .
echo $SONIC_DOCKER_REGISTRY/$SONIC_CONTAINER_NAME:v${n}

# Now run a local test. See https://contest.openai.com/details
python -m retro_contest run --agent $SONIC_DOCKER_REGISTRY/$SONIC_CONTAINER_NAME:v${n} \
    --results-dir results --no-nv --use-host-data \
    SonicTheHedgehog-Genesis GreenHillZone.Act1

docker push $SONIC_DOCKER_REGISTRY/$SONIC_CONTAINER_NAME:v${n}
