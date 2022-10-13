#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy unifyai/ivy:latest python3 -m pytest ivy_tests/test_first_ivy_model.py -s
