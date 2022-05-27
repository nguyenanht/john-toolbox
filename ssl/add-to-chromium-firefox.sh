#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux
  certificateFile="local-CA-johntoolbox-signed.crt"
  certificateName="Local-johntoolbox"
  for certDB in $(find  ~/.mozilla* ~/.pki -name "cert9.db")
  do
    certDir=$(dirname ${certDB});
    #log "mozilla certificate" "install '${certificateName}' in ${certDir}"
    certutil -D -n "${certificateName}" -d sql:${certDir}
    certutil -A -n "${certificateName}" -t "TCu,Cuw,Tuw" -i ${certificateFile} -d sql:${certDir}
  done
fi
