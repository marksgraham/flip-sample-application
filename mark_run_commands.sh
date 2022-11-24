docker run --rm -it -v /home/mark/projects/federated_learning/flip-sample-application/data:/data --entrypoint bash nvflare-in-one

bash start_nvflare_components.sh &
./poc/admin/startup/fl_admin.sh

upload_app flip-app
set_run_number 1
deploy_app flip-app all
start_app all

