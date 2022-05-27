#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    brew install openssl
    brew install mkcert
    brew install nss 
else
    sudo apt install libnss3-tools openssl -y
fi


if [[ ! "$(openssl version)" == *"1.1.1"* ]]; then
    echo "Upgrading OpenSSL to version 1.1.1"
    sudo apt install libnss3-tools libssl1.0.0 libssl-dev -y
    wget https://www.openssl.org/source/openssl-1.1.1c.tar.gz
    tar xzvf openssl-1.1.1c.tar.gz
    cd openssl-1.1.1c
    ./config
    make -j8
    sudo make install
    cd ..
    rm -rf openssl-1.1.1c/ openssl-1.1.1c.tar.gz
    sudo apt install openssl libnss3-tools libssl1.0.0 libssl-dev -y --reinstall
    echo "New version: $(openssl version)"
    else
        echo "Already having OpenSSL with version 1.1.1"
fi