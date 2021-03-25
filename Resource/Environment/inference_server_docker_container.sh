# Configurable variables.
image=hasush/mnist_classifier_inference_server:0p1
port=5001
local_dir_map=/home/hasush

# Run the container.
xhost +local:
docker run  \
    --gpus all \
    --net=host \
    -p 5000:5000 \
    -p 5001:5001 \
    -it  \
    --rm  \
    -v /tmp/.X11-unix:/tmp/.X11-unix  \
    -e DISPLAY=$DISPLAY  \
        --env QT_X11_NO_MITSHM=1 \
    -v ${local_dir_map}:${local_dir_map} \
    ${image}