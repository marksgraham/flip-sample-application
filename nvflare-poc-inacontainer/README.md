# nvflare_all_in_one_container

This is a dockerfile that will create a container with a NVFlare Server and two containers and start them.
Copy any NVFlare to ``/apps`` directory, the dockerfile will copy applications in this folder to the transfer section of NVFlare Admin.


To run either use Jetbrains Docker Service plugin (Recommended)

OR

Use the docker build commands

``docker build -t xnat-in-one .``
``docker run xnat-in-one``

If you exec into the container you should be able to run ``fl-admin.sh``

``docker exec -it <name> bash``

If your using CIFAS you will need to disable downloads otherwise it will try and download the CIFAS dataset which has problem from within a container.
All NVFlare logs are printed to stdout, which means you can view them by using the docker logs command

``docker logs <name>``

