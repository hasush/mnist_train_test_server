# mnist_train_test_server

Train and evaluate an MNIST classifier. 

Use flask to deploy a launch server which contacts an inference server which deploy the MNIST model to perform inference on a subset of the MNIST test set and returns results back to the launch server.

GPU compatibilitiy contingent on hardware driver compatibility with torch 1.6.

### How to use.

## Environment
Configure environment using requirements in Dockerfile located at Resource/Environments/Dockerfile or pull docker images from public docker hub:

ASDF_INSERT_LAUNCH_SERVER
ASDF_INSERT_INFERENCE_SERVER

Can launch docker images with following commands located at `Resource/Environments`:
inference_server_docker_container.sh  launch_server_docker_container.sh
