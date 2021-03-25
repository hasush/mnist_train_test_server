# Train an MNIST classifier using pytorch and then deploy the model to flask containers in seperate docker images.

Train and evaluate an MNIST classifier. 

Use flask to deploy a launch server which contacts an inference server which deploy the MNIST model to perform inference on a subset of the MNIST test set and returns results back to the launch server.

GPU compatibilitiy contingent on hardware driver compatibility with torch 1.6. Be sure to set `gpu` flag  in `config.py` to `False` if having any problems with gpu.

## How to use

### Environment
Configure environment using requirements in Dockerfile located at `Resource/Environments/Dockerfile` or pull docker images from public docker hub:

https://hub.docker.com/repository/docker/hasush/mnist_classifier_launch_server

https://hub.docker.com/repository/docker/hasush/mnist_classifier_inference_server

Can launch docker images with following commands located at `Resource/Environments`:

`inference_server_docker_container.sh`  `launch_server_docker_container.sh`

Be sure to configure `local_dir_map` in the `*_docker_container.sh` file to point to a directory on your hard drive to where you have cloned this repo.

Be sure port 5000 and 5001 are free on your device.

After launching container, please export the mnist_classifier module to the `PYTHONPATH`.

`export PYTHONPATH=local_dir_map/path/to/mnist_train_test_server`

Example:

In the files `inference_server_docker_container.sh` and `launch_server_docker_container.sh`, make `local_dir_map=/home/your_user_id`.

Launch the bash files to deploy the docker images.

If you cloned this repo to `/home/your_user_id/Documents/mnist_train_test_server`, then with 'user_id'='hasush', add the mnist_train_test_server module to the pythonpath by issuing the command.

```
hasush@wardell:~/Documents$ export PYTHONPATH=/home/hasush/Documents/mnist_train_test_server/Source/
```
### Train

Make sure that the fields of configuration file `config.py`: `image_dir` and `labels_dir` are set to the path of the MNIST data.
Set the `model_training_checkpoint_path` field in the configuration file is set to where you want to save the model.
Likewise, change any other parameters in the configuration file.

Train the model by issuing the driver with the train option.
```
root@wardell:/home/hasush/Documents/mnist_train_test_server/Source/mnist_classifier# python3 driver.py --mode train
```

### Eval

Make sure that the field of configuration file `config.py`: `model_evaluate_checkpoint_path` is set to where you want to load the model.

Test the model by issuing the driver with the test option.
```
root@wardell:/home/hasush/Documents/mnist_train_test_server/Source/mnist_classifier# python3 driver.py --mode test
```

### Deploy the launch server
Deploy a flask server which creates an endpoint where when visited at `0.0.0.0:5000/launch_inference` communicates with another flask server running at `0.0.0.0:5001/run_inference` (this can be in many environments including two separate docker containers) to initiate the other's server's remote procedure call to run inference on the MNIST dataset.

To deploy the server:
```
root@wardell:/home/hasush/Documents/mnist_train_test_server/Source/mnist_classifier# python3 driver.py --mode launch_server
 * Serving Flask app "mnist_classifier.deploy_launch_server" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

To initiate communication with the other server, visit endpoint at: `http://localhost:5000/launch_inference`. Note, other server must be up and running.

### Deploy the inference server.
Deploy a flask server which creates an endpoint where when visited at `0.0.0.0:5001/run_inference` launches inference on the MNIST test data set given the model pointed to in the configuration file.

Make sure that the field of configuration file `config.py`: `model_evaluate_checkpoint_path` is set to where you want to load the model.

To deploy the server:
```
root@wardell:/home/hasush/Documents/mnist_train_test_server/Source/mnist_classifier# python3 driver.py --mode inference_server
 * Serving Flask app "mnist_classifier.deploy_inference_server" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5001/ (Press CTRL+C to quit)
```

To initiate communication with the other server, visit endpoint at: `http://localhost:5001/run_inference`.

### Output
If everything is running correctly, one should be able to visit http://localhost:5001/run_inference or http://localhost:5000/launch_inference and produce the outputs shown in the following images:
https://github.com/hasush/mnist_train_test_server/blob/main/Resource/Images/inference_server.png
https://github.com/hasush/mnist_train_test_server/blob/main/Resource/Images/launch_server.png
