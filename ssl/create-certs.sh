#!/bin/bash
# https://www.wikihow.com/Be-Your-Own-Certificate-Authority
# 1 Create a private key
# 2 Create a certificate signing request.
rm -rf cert-local*


OLDIFS=$IFS; IFS='|'; for i in \
    'johntoolbox|CN=*.johntoolbox.localhost|DNS:*.localhost,DNS:*.johntoolbox.localhost,DNS:johntoolbox.localhost'
do
    set -- $i;
    # Create CSR (Certificate Signature Request)
    openssl req \
        -verbose \
        -nodes \
        -newkey rsa:4096 \
        -keyout cert-local-$1.key \
        -out cert-local-$1.csr \
        -subj "/C=FR/ST=Ile-de-France/L=Paris/O=johntoolbox/OU=johntoolbox Team/$2/emailAddress=contact@nguyenjohnathan.com" \
        -reqexts SAN \
        -config <(cat /etc/ssl/openssl.cnf <(printf "\n[SAN]\nsubjectAltName=$3"))

    # Sign it using own local CA
    openssl ca \
        -extensions SAN \
        -md sha256 \
        -days 824 \
        -rand_serial \
        -cert local-CA-$1-signed.crt \
        -out cert-local-$1.crt \
        -keyfile local-CA-$1.key \
        -config <(cat /etc/ssl/openssl.cnf <(printf "\n[SAN]\nsubjectAltName=$3")) \
        -infiles cert-local-$1.csr

    openssl rsa \
        -in cert-local-$1.key \
        -out cert-local-$1.unsecured.key

done; IFS=$OLDIFS
