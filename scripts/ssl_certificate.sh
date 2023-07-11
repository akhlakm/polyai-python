#!/bin/sh

# Steps for SSL connection. Repeat when expired.
# 1. Generate a local certificate authority.
# 2. Install the CA .crt file in your client devices/docker containers/browsers.
# 3. Generate SSL key pairs using the CA.
# 4. Upload the signed SSL files (e.g. use rsync) to the server.

local_ca() {
    # Create a local certificate authority.
    mkdir -p ~/.certs
    echo "Creating key file ..."
    openssl genrsa -des3 -out ~/.certs/my_ssl_CA.key 4096     # private key file
    echo "Creating certificate ..."
    openssl req -x509 -new -nodes -key ~/.certs/my_ssl_CA.key -sha256 -days 30 -out ~/.certs/my_ssl_CA.crt  # public certificate
    echo "CA generated. The certificate file must be installed on the user devices."
    echo "The key file should be securely stored and used to sign key pairs."
}

install_ca() {
    # Install the CA certificate on Ubuntu
    # For other devices, see https://deliciousbrains.com/ssl-certificate-authority-for-local-https-development/ 
    sudo apt-get install ca-certificates
    sudo cp ~/.certs/my_ssl_CA.crt /usr/local/share/ca-certificates/my_ssl_CA.crt || exit 10
    sudo update-ca-certificates
}

ssl_keys() {
    [ -f ~/.certs/my_ssl_CA.crt ] || exit 10
    mkdir -p keys
    echo "Creating key file ..."
    openssl genrsa -out keys/ssl.key 4096    # private key file
    echo "Creating CA request files ..."
    openssl req -new -key keys/ssl.key -out keys/ssl.csr  # CA request file

    # DNS cannot be ip address.
    cat > keys/ssl.ext << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
DNS.2 = polyai
DNS.3 = tunnel
EOF

    echo "Creating certificate ..."
    openssl x509    -req -in keys/ssl.csr -CA ~/.certs/my_ssl_CA.crt \
                    -CAkey ~/.certs/my_ssl_CA.key -CAcreateserial \
                    -out keys/ssl.crt -days 30 -sha256 -extfile keys/ssl.ext || exit 20

    echo "Done! Use the crt and key files for SSL server on ssl."
    echo "Note! The CA certificate (~/.certs/my_ssl_CA.crt) must be installed on the client devices."
}

view_cert() {
    openssl x509 -text < $1
}

"$@"
