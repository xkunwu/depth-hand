#!/bin/bash

jupyter-notebook list  | sed -n "s/^.*token=\(\S\+\).*$/\1/p"
