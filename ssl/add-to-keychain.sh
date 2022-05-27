# This script will automatically add the certicicate in Mac os X Keychain
# It only runs if you are on a Mac
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "On 🍎: automatically add certicifate to " local-CA-johntoolbox-signed.crt
    sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain local-CA-johntoolbox-signed.crt
fi