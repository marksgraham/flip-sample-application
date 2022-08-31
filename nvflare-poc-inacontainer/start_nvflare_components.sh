#!/bin/bash

./poc/server/startup/start.sh &
./poc/site-1/startup/start.sh &
./poc/site-2/startup/start.sh &

sleep infinity