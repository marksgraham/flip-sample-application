# nvflare_all_in_one_container

This is a dockerfile that will create a container with a NVFlare Server and two containers and start them.
Copy any NVFlare Applications you wish to use to the ``/apps`` directory, the dockerfile will copy applications in this folder
to the transfer section of the NVFlare Admin application.

### 1. Build Docker Image & Run Docker Container

Use the docker build commands and run the container

``docker build . -t nvflare-in-one``

``docker run nvflare-in-one``

### 2. Execute & Monitor NVFlare

If you exec into the container

``docker exec -it <name> bash``

You will be able to run ``fl-admin.sh``
The username and password for this container are ``admin``
This will give you access to all the NVFlare Admin commands

To start the flip-app bundled within this repository:
```
set_run_number 1

upload_app flip-app

deploy_app flip-app all

start_app all
```

The logs for the model execution are written to STDOUT and can be accessible
by viewing the logs of the container

``docker logs <name>``

### 3. Inserting Files into the container

If you wish to test the utillization of resources in your model E.g. DICOMS 
I recommend using the copy commnd to copy the files to the container

```
docker cp <path_to_your_resource> <container_name>:/dir
```

You can then edit the flip.py module and change the response to the parent directory of where your files were copied to
e.g. ``/dir``

Within FLIP the files are identifiable by the accession number, as returned within the dataframe

``/dir/<accession_number>``
