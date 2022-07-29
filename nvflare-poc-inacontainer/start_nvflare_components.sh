#!/bin/bash

./poc/server/startup/start.sh &
./poc/site-1/startup/start.sh localhost TRUST-A &
./poc/site-2/startup/start.sh localhost TRUST-B &

sleep infinity