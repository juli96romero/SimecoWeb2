from django.urls import re_path 
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/socket-server/', consumers.ChatConsumer.as_asgi()),
    re_path(r'ws/socket-server-image/', consumers.ImageConsumer.as_asgi()),
    re_path(r'ws/socket-saver-image/', consumers.Outputer.as_asgi()),
    re_path(r'ws/socket-lineal/', consumers.GeneradorLineal.as_asgi()),
]