from polyai.api import requestor

MAX_TIMEOUT = 20

class APIEngine:
    def __init__(self, engine = None, **kwargs) -> None:
        pass

    @classmethod
    def class_url(cls, model):
        if hasattr(cls, '__object__'):
            return getattr(cls, '__object__').replace(".", "/")
        else:
            return "/"

    @classmethod
    def __prepare_create_request(cls, api_key=None, api_base=None,
                                 organization=None, **params):
        model = params.get("model", None)
        stream = params.get("stream", False)
        headers = params.pop("headers", None)
        timeout = params.pop("request_timeout", None)

        # # validate
        # if model is None:
        #     raise error.InvalidRequestError(
        #         "Must provide a 'model' parameter to create a %s" % cls)
        
        req = requestor.APIRequestor(api_key, api_base, organization)
        url = cls.class_url(model)

        return stream, headers, timeout, req, url, params

    @classmethod
    def create(cls, api_key=None, api_base=None, request_id=None,
               organization=None, **params):
        
        stream, headers, timeout, req, url, params = cls.__prepare_create_request(
            api_key, api_base, organization, **params)

        response, _, api_key = req.request("post", url,
            params=params,
            headers=headers,
            stream=stream,
            request_id=request_id,
            request_timeout=timeout,
        )

        if stream:
            return (line for line in response)
        else:
            return response

