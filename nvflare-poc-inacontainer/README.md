# nvflare_all_in_one_container

This is a dockerfile that will create a container with a NVFlare Server and two containers and start them.
Copy any NVFlare to ``/apps`` directory, the dockerfile will copy applications in this folder to the transfer section of NVFlare Admin.

### 1. Build Docker Container

Use the docker build commands and run the container

``docker build -t xnat-in-one ``

``docker run xnat-in-one``

### 2. Run Docker Container

If you exec into the container you should be able to run ``fl-admin.sh``
This will give you access to all the NVFlare Admin commands


``docker exec -it <name> bash``
``docker logs <name>``

