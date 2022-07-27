# nvflare_all_in_one_container
This is nvflare all in one container.
This is a dockerfile that will create a container with a NVFlare Server and two containers.
To run either use Jetbrains Docker Service plugin (Recommended)
OR
Use the docker build commands
``docker build -t xnat-in-one .``
``docker run xnat-in-one``

If you exec into the container you should be able to run ``fl-admin.sh``

If your using CIFA you may also want to tell it not download the files as this has problems doing it all within 1 container on a local network.

Have fun!