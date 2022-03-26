## Run a docker container
cd ${catkin_ws}
docker run --gpus all -it --shm-size=8192m --privileged --network=host -v ${PWD}/src/gqcnn/:/home/docker/gqcnn_ws/src/gqcnn/:rw -v ${PWD}/src/task_msgs/:/home/docker/gqcnn_ws/src/task_msgs/:rw -v /tmp/.X11-unix/:/tmp/.X11-unix:rw --env="DISPLAY" --name=gqcnn_focal gqcnn/focal bash