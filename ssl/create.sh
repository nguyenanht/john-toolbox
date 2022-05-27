#!/bin/bash
chmod 777 -R .
apt-get update -y
apt-get install make sudo -y
./install-prerequisites.sh
./create-ca.sh
./create-certs.sh
chmod 777 -R .
