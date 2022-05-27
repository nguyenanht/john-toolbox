#!/bin/bash
# https://www.wikihow.com/Be-Your-Own-Certificate-Authority
# 1 Generate your CA's private key by issuing the following command
# 2 Create a certificate signing request. 
# openssl genrsa -verbose -des3 -out local-CA.key 4096
# openssl req -verbose -new -key server.CA.key -out server.CA.csr -sha256
rm -rf demoCA *.csr *.key *.crt *.pem
mkdir demoCA/newcerts -p
touch demoCA/index.txt demoCA/index.txt.attr demoCA/serial

# echo 4096 > demoCA/serial
openssl req \
    -verbose \
    -nodes \
    -newkey rsa:4096 \
    -keyout local-CA-johntoolbox.key \
    -out local-CA-johntoolbox.csr \
    -subj "/C=FR/ST=Ile-de-France/L=Paris/O=johntoolbox/OU=johntoolbox Team/CN=johntoolbox Local-johntoolbox Certificates/emailAddress=contact@nguyenjohnathan.com"
# 4 Self-sign your certificate
# openssl ca -extensions v3_ca -out server.CA-signed.crt -keyfile server.CA.key -verbose -selfsign -md sha256 -enddate 330630235959Z -infiles local-CA.csr
openssl ca \
    -rand_serial \
    -out local-CA-johntoolbox-signed.crt \
    -keyfile local-CA-johntoolbox.key \
    -verbose \
    -selfsign \
    -md sha256 \
    -days 824 \
    -extensions v3_ca \
    -infiles local-CA-johntoolbox.csr
        # -extensions v3_ca \ 

# 5 Inspect your CA certificate
# openssl x509 -noout -text -in local-CA-signed.crt

cat local-CA-johntoolbox.key local-CA-johntoolbox-signed.crt > local-CA-johntoolbox.pem
