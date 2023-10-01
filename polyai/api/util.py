import os
import polyai.api
from urllib.parse import urlparse


def default_api_key() -> str:
    if polyai.api.api_key_path:
        with open(polyai.api.api_key_path, "rt") as k:
            api_key = k.read().strip()
            if not api_key.startswith("pl-"):
                raise ValueError(f"Malformed API key in {polyai.api.api_key_path}.")
            return api_key
    elif polyai.api.api_key is not None:
        return polyai.api.api_key
    else:
        raise polyai.api.error.AuthenticationError(
            "No API key provided. You can set your API key in code using 'polyai.api_key = <API-KEY>', or you can set the environment variable POLYAI_API_KEY=<API-KEY>). If your API key is stored in a file, you can point the polyai module at it with 'polyai.api_key_path = <PATH>'."
        )


def create_ssh_tunnel():
    """
    Update the API endpoint URL to connect via SSH tunnel.
    
    SSH_TUNNEL_HOST, SSH_USERNAME and SSH_PASSWORD environment
    variables must be defined and loaded, e.g. using the dotenv package.

    Returns:
        SSHTunnelForwarder instance.
    """
    import polyai.sett as sett
    from sshtunnel import SSHTunnelForwarder

    host = sett.API.ssh_tunnel_host
    port = sett.API.ssh_tunnel_port
    usernm = sett.API.ssh_tunnel_user
    passwd = sett.API.ssh_tunnel_pass

    if host is None or usernm is None:
        print("Not creating SSH tunnel.")
        print("To use SSH tunnel, make sure SSH_TUNNEL_HOST, "
              "SSH_USERNAME and SSH_PASSWORD environment "
              "variables are correctly defined and loaded.")
        return None

    # API endpoint
    endpoint_host = urlparse(sett.API.polyai_api_base).hostname
    endpoint_port = urlparse(sett.API.polyai_api_base).port
    endpoint_path = urlparse(sett.API.polyai_api_base).path

    server = SSHTunnelForwarder(
        (
            host, int(port)
        ),
        ssh_username=usernm,
        ssh_password=passwd,
        remote_bind_address=(endpoint_host, int(endpoint_port)),
    )

    server.start()
    polyai.api.api_base = f"http://{server.local_bind_host}:{server.local_bind_port}{endpoint_path}"

    return server
