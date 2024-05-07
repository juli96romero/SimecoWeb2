from django.shortcuts import render
from django.http import HttpResponse
from .red import main  # Corrected import statement


def my_view(request):
    # esto llama a la red
    #return render(request, main)

    #esto llama al frontend
    return render(request, 'api/main_page.html')
