import polyai.api

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

