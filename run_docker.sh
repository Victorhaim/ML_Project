#!/bin/bash
# Script to run 3L-Cache docker container with appropriate mounts

docker run -v /Users/victor_haim/Documents/technion/AI_PROJECT/ML_Project/data:/3L-Cache/data \
           -v /Users/victor_haim/Documents/technion/AI_PROJECT/ML_Project/3LCache:/3L-Cache/3LCache \
           -it 3lcache bash
