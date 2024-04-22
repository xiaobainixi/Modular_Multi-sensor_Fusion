nvidia-docker run --privileged -it -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
    --volume=/home:/home  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --net=host --ipc=host --shm-size=1gb \
    --name=modular_fusion_2 --env="DISPLAY=$DISPLAY"  fusion:1.0 /bin/bash