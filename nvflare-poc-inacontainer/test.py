#!/usr/bin/env python3

"""Connects to the NVFlare POC Server, starts the hello-numpy-sag
example application and verifies that the FL run is complete. Verifies
the FL run is complete by asserting that the end product of the FL run -
server.npy - is visible. The run number is deleted and recreated to ensure that
server.npy is created anew before it is asserted."""

import sys
import time
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI

api = FLAdminAPI(
    host="localhost",
    port=8003,
    # ca_cert="rootCA.pem",
    # client_cert="client.crt",
    # client_key="client.key",
    upload_dir="transfer",
    download_dir="transfer",
    poc=True,
    # debug=False
)


def wrapper(api_command_result):
    if api_command_result["status"] != "SUCCESS":
        raise RuntimeError(api_command_result["status"])
    return api_command_result


def main():
    wrapper(api.login_with_password("admin", "admin"))

    wrapper(api.upload_app("flip-app"))
    wrapper(api.delete_run_number(1))
    wrapper(api.set_run_number(1))
    wrapper(api.deploy_app("flip-app", "all"))
    wrapper(api.start_app("all"))

    status = wrapper(api.check_status("server"))
    while (status["details"]["server_engine_status"] != "stopped"):
        print("server engine status: " + status["details"]["server_engine_status"])
        print("FL still in progress. Waiting for FL run to finish.")
        print("will check server engine status again soon...")
        time.sleep(3)
        status = wrapper(api.check_status("server"))
    print("server engine has finished running application")

    result = wrapper(api.ls_target("server", "-R"))
    result = result["details"]["message"]
    if "./run_1/app_server:\nFL_global_model.pt" not in result:
        raise RuntimeError("Aggregated model not found. Something went wrong.")
    else:
        print("Aggregated model found. FL run was successful!")

    result_trust_a = wrapper(api.cat_target(target="TRUST-A", file="log.txt"))
    result_trust_b = wrapper(api.cat_target(target="TRUST-A", file="log.txt"))
    result_server = wrapper(api.cat_target(target="server", file="log.txt"))

    if "ERROR" in result_trust_a["details"]["message"]:
        raise RuntimeError("ERROR reported on Trust A")
    if "ERROR" in result_trust_b["details"]["message"]:
        raise RuntimeError("ERROR reported on Trust B")
    if "ERROR" in result_server["details"]["message"]:
        raise RuntimeError("ERROR reported on Server")

    sys.exit(0)


if __name__ == "__main__":
    main()
