# Train an MNIST classifier using pytorch and then deploy the model to flask containers in seperate docker images.

Train and evaluate an MNIST classifier. 

Use flask to deploy a launch server which contacts an inference server which deploy the MNIST model to perform inference on a subset of the MNIST test set and returns results back to the launch server.

GPU compatibilitiy contingent on hardware driver compatibility with torch 1.6.

## How to use.

### Environment
Configure environment using requirements in Dockerfile located at `Resource/Environments/Dockerfile` or pull docker images from public docker hub:

ASDF_INSERT_LAUNCH_SERVER
ASDF_INSERT_INFERENCE_SERVER

Can launch docker images with following commands located at `Resource/Environments`:

`inference_server_docker_container.sh`  `launch_server_docker_container.sh`

Be sure to configure `local_dir_map` in the `*_docker_container.sh` file to point to a directory on your hard drive to where you have cloned this repo.

Be sure port 5000 and 5001 are free on your device.

After launching container, please export the mnist_classifier module to the `PYTHONPATH`.

`export PYTHONPATH=local_dir_map/path/to/mnist_train_test_server`

Example:
In the files `inference_server_docker_container.sh` and `launch_server_docker_container.sh`, make `local_dir_map=/home/your_user_id`.

Launch the bash files to deploy the docker images.
### Train

