# core/views.py

from django.shortcuts import render , redirect , HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required

def home(request):
    """View for the Main Page (Home)"""
    return render(request, 'core/index.html')
def signup_view(request):
    print("Signup view triggered with method:", request.method)
    if request.method == 'POST':
        print("POST data:", request.POST)
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(f"Data received: {name}, {email}, {password}")

        if User.objects.filter(username=email).exists():
            print("User already exists!")
            messages.error(request, 'Email already registered!')
            return redirect('signup')

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=name
        )
        user.save()
        print("User created successfully!")
        messages.success(request, 'Account created successfully! Please sign in.')
        print("✅ Redirecting to signin...")
        return redirect('signin')

    return render(request, 'core/signup.html')

def signin_view(request):
   if request.method == 'POST':
        print("Signin POST triggered")
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(f"Email: {email}, Password: {password}")

        user = authenticate(username=email, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome, {user.first_name}!')
            return redirect('home')
        else:
            messages.error(request, 'Invalid email or password.')
            return redirect('signin')

   return render(request, 'core/signin.html')
def signout_view(request):
    """Handle user sign out"""
    logout(request)
    messages.info(request, 'You have been signed out successfully!')
    return redirect('signin')
def about(request): 
    """View for the About Page"""
    return render(request, 'core/about.html')
def market_data(request):
    """View for the Market Data Page"""
    return render(request, 'core/market_data.html')
def privacy_policy(request):
    """View for the Privacy Policy Page"""
    return render(request, 'core/privacy_policy.html',{})
def terms_of_service(request):
    """View for the Terms of Service Page"""    
    return render(request, 'core/terms_of_service.html',{})
@login_required(login_url='/signin/')
def analysis(request):
    # return HttpResponse("Analysis Page executed")
    return render(request, 'core/analysis.html', {})

