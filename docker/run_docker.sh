docker run -it \
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home:/home \
--gpus all \
--name modular_fusion \
fusion:1.0 \
/bin/bash