from django.urls import re_path 
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/socket-principal128/', consumers.Principal128.as_asgi()),
    re_path(r'ws/socket-principal256/', consumers.Principal256.as_asgi()),
    re_path(r'ws/socket-principal-front/', consumers.Socket_Principal_FrontEnd.as_asgi()),
    re_path(r'ws/socket-server-image/', consumers.ImageConsumer.as_asgi()),
    re_path(r'ws/socket-pickle/', consumers.PickleHandler.as_asgi()),
    re_path(r'ws/brightness/', consumers.Brightness.as_asgi()), 
    re_path(r'ws/socket-combinedSlice/', consumers.combinedSlice.as_asgi()),
]