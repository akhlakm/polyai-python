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

    from sshtunnel import SSHTunnelForwarder

    host = os.environ.get("SSH_TUNNEL_HOST")
    usernm = os.environ.get("SSH_USERNAME")
    passwd = os.environ.get("SSH_PASSWORD")

    if host is None or usernm is None:
        print("Not creating SSH tunnel.")
        print("To use SSH tunnel, make sure SSH_TUNNEL_HOST, "
              "SSH_USERNAME and SSH_PASSWORD environment "
              "variables are correctly defined and loaded.")
        return None

    # API endpoint
    llm_host = os.environ.get("LLM_HOST", None)
    llm_port = os.environ.get("LLM_PORT", None)

    if llm_host is None:
        llm_host = urlparse(polyai.api.api_base).hostname
    if llm_port is None:
        llm_port = urlparse(polyai.api.api_base).port

    server = SSHTunnelForwarder(
        (
            host,
            int(os.environ.get("SSH_TUNNEL_PORT") or 22)
        ),
        ssh_username=usernm,
        ssh_password=passwd,
        remote_bind_address=(llm_host, int(llm_port)),
    )

    server.start()
    polyai.api.api_base = f"http://{server.local_bind_host}:{server.local_bind_port}/api/v1/"

    return server
