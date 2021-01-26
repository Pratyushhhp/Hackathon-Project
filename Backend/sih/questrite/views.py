from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from .models import Personality
from .models import School
from .models import Graduates
from .models import Dropouts

# Create your views here.

def index(request):
    return render(request, 'index.html')

def signup(request):

    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:

            if User.objects.filter(username = username).exists():
                messages.info(request, 'username already taken')
                return redirect('signup')

            elif User.objects.filter(email = email).exists():
                messages.info(request, 'email already taken')
                return redirect('signup')            
            else:
                user = User.objects.create_user(username = username, password = password1, email = email, first_name = first_name, last_name = last_name)
                user.save()
                return redirect ('about')

        else:
            messages.info(request, 'password not matching')
            return redirect('signup')

        return redirect('/')

    else:
        return render(request, 'signup.html')


def logout(request):
    auth.logout(request)
    return redirect('/')

    

def about(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username = username, password = password)

        if user is not None:
            auth.login(request, user)

            return redirect("/")

        else:
            messages.info(request, "invalid credentials")
            return redirect('about')

    else:
        return render(request, 'about.html')

def analyst(request):
    return render(request, 'analyst.html')

def contactus(request):
    return render(request, 'contactus.html')

def faqs(request):
    return render(request, 'faqs.html')

def ourservices(request):
    return render(request, 'ourservices.html')

def personality(request):
    return render(request, 'personality.html')

def result(request):
    return render(request, 'result.html')

def resultp(request):
    return render(request, 'resultp.html')

def reviews(request):
    return render(request, 'reviews.html')

def test1(request):
    return render(request, 'test1.html')

def test2(request):
    return render(request, 'test2.html')

def test3(request):
    return render(request, 'test3.html')

def realistic(request):
    return render(request, 'realistic.html')

def investigative(request):
    return render(request, 'investigative.html')

def artistic(request):
    return render(request, 'artistic.html')

def social(request):
    return render(request, 'social.html')

def enterprising(request):
    return render(request, 'enterprising.html')

def conventional(request):
    return render(request, 'conventional.html')

def counsellor(request):
    return render(request, 'counsellor.html')

def scholarship(request):
    return render(request, 'scholarship.html')

def mcq1(request):
    d = 0
    q = 0
    v = 0
    se = 0
    ss = 0
    sp = 0

    if request.method == 'POST':

        q1 = request.POST['GROUP1']
        if q1 == 'a':
            d = d + 1


        q2 = request.POST['GROUP2']
        if q2 == 'b':
            d = d + 1

        q3 = request.POST['GROUP3']
        if q3 == 'a':
            d = d + 1

        q4 = request.POST['GROUP4']
        if q4 == 'b':
            d = d + 1

        q5 = request.POST['GROUP5']
        if q5 == 'a':
            d = d + 1

        q6 = request.POST['GROUP6']
        if q6 == 'a':
            q = q + 1

        q7 = request.POST['GROUP7']
        if q7 == 'd':
            q = q + 1

        q8 = request.POST['GROUP8']
        if q8 == 'a':
            q = q + 1

        q9 = request.POST['GROUP9']
        if q9 == 'a':
            q = q + 1

        q10 = request.POST['GROUP10']
        if q10 == 'b':
            q = q + 1

        q11 = request.POST['GROUP11']
        if q11 == 'a':
            v = v + 1

        q12 = request.POST['GROUP12']
        if q12 == 'c':
            v = v + 1


        q13 = request.POST['GROUP13']
        if q13 == 'b':
            v = v + 1

        q14 = request.POST['GROUP14']
        if q14 == 'd':
            v = v + 1

        q15 = request.POST['GROUP15']
        if q15 == 'c':
            v = v + 1

        q16 = request.POST['GROUP16']
        if q16 == 'c':
            se = se + 1

        q17 = request.POST['GROUP17']
        if q17 == 'd':
            se = se + 1

        q18 = request.POST['GROUP18']
        if q18 == 'a':
            se = se + 1

        q19 = request.POST['GROUP19']
        if q19 == 'd':
            se = se + 1

        q20 = request.POST['GROUP20']
        if q20 == 'b':
            se = se + 1

        q21 = request.POST['GROUP21']
        if q21 == 'c':
            ss = ss + 1

        q22 = request.POST['GROUP22']
        if q22 == 'c':
            ss = ss + 1

        q23 = request.POST['GROUP23']
        if q23 == 'b':
            ss = ss + 1

        q24 = request.POST['GROUP24']
        if q24 == 'd':
            ss = ss + 1

        q25 = request.POST['GROUP25']
        if q25 == 'a':
            ss = ss + 1

        q26 = request.POST['GROUP26']
        if q26 == 'a':
            sp = sp + 1
        

        q27 = request.POST['GROUP27']
        if q27 == 'a':
            sp = sp + 1

        q28 = request.POST['GROUP28']
        if q28 == 'a':
            sp = sp + 1

        q29 = request.POST['GROUP29']
        if q29 == 'b':
            sp = sp + 1

        q30 = request.POST['GROUP30']
        if q30 == 'd':
            sp = sp + 1


        maximum =  max(d,q,v,se,sp,ss)

        if maximum == d:
            message = "Conventional"
            message1 = "Accounting, Computer Information System, Library, Science and Administrative Work"

        elif maximum == q:
            message = "Investigative"
            message1 = "Biology, Chemistry, Computer Science, Economics, Law, Mathematics, Psychology and Data Analyst"

        elif maximum == v:
            message = "Artistic"
            message1 = "Advertising, Art, Graphic Design, Music, Theater and Writer"

        elif maximum == se:
            message = "Enterprising"
            message1 = "Business, Finance, Law Enforcement, Real Estate, Marketing and Journalism"

        elif maximum == ss:
            message = "Social"
            message1 = "Communications, Anthropology, Education, Nursing, Religion, Sociology and Social Work "

        elif maximum == sp:
            message = "Realistic"        
            message1 = "Engineers, Architecture, Dietitian, Criminal Justice, Forestry and Athelete"        

        return render(request, 'result.html', {'message': message, 'message1': message1})
    else:    
       return render(request, 'mcq1.html')


def mcq2(request):
    d = 0
    q = 0
    v = 0
    se = 0
    ss = 0
    sp = 0

    if request.method == 'POST':

        q1 = request.POST['GROUP1']
        if q1 == 'b':
            d = d + 1


        q2 = request.POST['GROUP2']
        if q2 == 'b':
            d = d + 1

        q3 = request.POST['GROUP3']
        if q3 == 'd':
            d = d + 1

        q4 = request.POST['GROUP4']
        if q4 == 'a':
            d = d + 1

        q5 = request.POST['GROUP5']
        if q5 == 'c':
            d = d + 1

        q6 = request.POST['GROUP6']
        if q6 == 'd':
            q = q + 1

        q7 = request.POST['GROUP7']
        if q7 == 'c':
            q = q + 1

        q8 = request.POST['GROUP8']
        if q8 == 'b':
            q = q + 1

        q9 = request.POST['GROUP9']
        if q9 == 'c':
            q = q + 1

        q10 = request.POST['GROUP10']
        if q10 == 'd':
            q = q + 1

        q11 = request.POST['GROUP11']
        if q11 == 'c':
            v = v + 1

        q12 = request.POST['GROUP12']
        if q12 == 'c':
            v = v + 1


        q13 = request.POST['GROUP13']
        if q13 == 'c':
            v = v + 1

        q14 = request.POST['GROUP14']
        if q14 == 'a':
            v = v + 1

        q15 = request.POST['GROUP15']
        if q15 == 'd':
            v = v + 1

        q16 = request.POST['GROUP16']
        if q16 == 'a':
            se = se + 1

        q17 = request.POST['GROUP17']
        if q17 == 'd':
            se = se + 1

        q18 = request.POST['GROUP18']
        if q18 == 'd':
            se = se + 1

        q19 = request.POST['GROUP19']
        if q19 == 'a':
            se = se + 1

        q20 = request.POST['GROUP20']
        if q20 == 'c':
            se = se + 1

        q21 = request.POST['GROUP21']
        if q21 == 'b':
            ss = ss + 1

        q22 = request.POST['GROUP22']
        if q22 == 'd':
            ss = ss + 1

        q23 = request.POST['GROUP23']
        if q23 == 'a':
            ss = ss + 1

        q24 = request.POST['GROUP24']
        if q24 == 'b':
            ss = ss + 1

        q25 = request.POST['GROUP25']
        if q25 == 'd':
            ss = ss + 1

        q26 = request.POST['GROUP26']
        if q26 == 'b':
            sp = sp + 1
        

        q27 = request.POST['GROUP27']
        if q27 == 'a':
            sp = sp + 1

        q28 = request.POST['GROUP28']
        if q28 == 'a':
            sp = sp + 1

        q29 = request.POST['GROUP29']
        if q29 == 'b':
            sp = sp + 1

        q30 = request.POST['GROUP30']
        if q30 == 'b':
            sp = sp + 1


        maximum =  max(d,q,v,se,sp,ss)

        if maximum == d:
            message = "Conventional"
            message1 = "Accounting, Computer Information System, Library, Science and Administrative Work"

        elif maximum == q:
            message = "Investigative"
            message1 = "Biology, Chemistry, Computer Science, Economics, Law, Mathematics, Psychology and Data Analyst"

        elif maximum == v:
            message = "Artistic"
            message1 = "Advertising, Art, Graphic Design, Music, Theater and Writer"

        elif maximum == se:
            message = "Enterprising"
            message1 = "Business, Finance, Law Enforcement, Real Estate, Marketing and Journalism"

        elif maximum == ss:
            message = "Social"
            message1 = "Communications, Anthropology, Education, Nursing, Religion, Sociology and Social Work "

        elif maximum == sp:
            message = "Realistic"        
            message1 = "Engineers, Architecture, Dietitian, Criminal Justice, Forestry and Athelete"        

        return render(request, 'result.html', {'message': message, 'message1': message1})
    else:    
       return render(request, 'mcq2.html')


def mcq3(request):
    d = 0
    q = 0
    v = 0
    se = 0
    ss = 0
    sp = 0

    if request.method == 'POST':

        q1 = request.POST['GROUP1']
        if q1 == 'a':
            d = d + 1


        q2 = request.POST['GROUP2']
        if q2 == 'b':
            d = d + 1

        q3 = request.POST['GROUP3']
        if q3 == 'a':
            d = d + 1

        q4 = request.POST['GROUP4']
        if q4 == 'a':
            d = d + 1

        q5 = request.POST['GROUP5']
        if q5 == 'a':
            d = d + 1

        q6 = request.POST['GROUP6']
        if q6 == 'c':
            q = q + 1

        q7 = request.POST['GROUP7']
        if q7 == 'd':
            q = q + 1

        q8 = request.POST['GROUP8']
        if q8 == 'a':
            q = q + 1

        q9 = request.POST['GROUP9']
        if q9 == 'c':
            q = q + 1

        q10 = request.POST['GROUP10']
        if q10 == 'b':
            q = q + 1

        q11 = request.POST['GROUP11']
        if q11 == 'c':
            v = v + 1

        q12 = request.POST['GROUP12']
        if q12 == 'd':
            v = v + 1


        q13 = request.POST['GROUP13']
        if q13 == 'b':
            v = v + 1

        q14 = request.POST['GROUP14']
        if q14 == 'c':
            v = v + 1

        q15 = request.POST['GROUP15']
        if q15 == 'c':
            v = v + 1

        q16 = request.POST['GROUP16']
        if q16 == 'c':
            se = se + 1

        q17 = request.POST['GROUP17']
        if q17 == 'a':
            se = se + 1

        q18 = request.POST['GROUP18']
        if q18 == 'd':
            se = se + 1

        q19 = request.POST['GROUP19']
        if q19 == 'b':
            se = se + 1

        q20 = request.POST['GROUP20']
        if q20 == 'a':
            se = se + 1

        q21 = request.POST['GROUP21']
        if q21 == 'c':
            ss = ss + 1

        q22 = request.POST['GROUP22']
        if q22 == 'c':
            ss = ss + 1

        q23 = request.POST['GROUP23']
        if q23 == 'b':
            ss = ss + 1

        q24 = request.POST['GROUP24']
        if q24 == 'd':
            ss = ss + 1

        q25 = request.POST['GROUP25']
        if q25 == 'd':
            ss = ss + 1

        q26 = request.POST['GROUP26']
        if q26 == 'a':
            sp = sp + 1
        

        q27 = request.POST['GROUP27']
        if q27 == 'a':
            sp = sp + 1

        q28 = request.POST['GROUP28']
        if q28 == 'a':
            sp = sp + 1

        q29 = request.POST['GROUP29']
        if q29 == 'b':
            sp = sp + 1

        q30 = request.POST['GROUP30']
        if q30 == 'b':
            sp = sp + 1


        maximum =  max(d,q,v,se,sp,ss)

        if maximum == d:
            message = "Conventional"
            message1 = "Accounting, Computer Information System, Library, Science and Administrative Work"

        elif maximum == q:
            message = "Investigative"
            message1 = "Biology, Chemistry, Computer Science, Economics, Law, Mathematics, Psychology and Data Analyst"

        elif maximum == v:
            message = "Artistic"
            message1 = "Advertising, Art, Graphic Design, Music, Theater and Writer"

        elif maximum == se:
            message = "Enterprising"
            message1 = "Business, Finance, Law Enforcement, Real Estate, Marketing and Journalism"

        elif maximum == ss:
            message = "Social"
            message1 = "Communications, Anthropology, Education, Nursing, Religion, Sociology and Social Work "

        elif maximum == sp:
            message = "Realistic"        
            message1 = "Engineers, Architecture, Dietitian, Criminal Justice, Forestry and Athelete"

        return render(request, 'result.html', {'message': message, 'message1': message1})
    else:    
       return render(request, 'mcq3.html')





def mcq4(request):
    r = 0
    i = 0
    a = 0
    s = 0
    e = 0
    c = 0

    if request.method == 'POST':
        q1 = request.POST['GROUP1']
        ans = Personality(questions = 1, answers = q1)
        ans.save()
        if q1 == '3':
            r = r + 1
        if q1 == '1':
            i = i+1

        if q1 == '1':
            a = a+1

        if q1 == '1':
            s = s+1

        if q1 == '3':
            e = e+1

        if q1 == '3':
            c = c+1


        q2 = request.POST['GROUP2']
        ans = Personality(questions = 2, answers = q2)
        ans.save()
        if q2 == 'a':
            r = r + 1
        
        if q2 == 'a':
            i = i+1

        if q2 == 'a':
            a = a+1

        if q2 == 'a':
            s = s+1

        if q2 == 'a':
            e = e+1

        if q2 == 'a':
            c = c+1

        q3 = request.POST['GROUP3']
        ans = Personality(questions = 3, answers = q3)
        ans.save()
        if q3 == 'a':
            r = r + 1
        
        if q3 == 'a':
            i = i+1

        if q3 == 'a':
            a = a+1

        if q3 == 'a':
            s = s+1

        if q3 == 'a':
            e = e+1

        if q3 == 'a':
            c = c+1


        q4 = request.POST['GROUP4']
        ans = Personality(questions = 4, answers = q4)
        ans.save()
        if q4 == 'd':
            r = r + 1
        
        if q4 == 'a':
            i = i+1

        if q4 == 'a':
            a = a+1

        if q4 == 'a':
            s = s+1

        if q4 == 'a':
            e = e+1

        if q1 == 'a':
            c = c+1


        q5 = request.POST['GROUP5']
        ans = Personality(questions = 5, answers = q5)
        ans.save()
        if q5 == 'a':
            r = r + 1

        if q5 == 'a':
            i = i+1

        if q5 == 'a':
            a = a+1

        if q5 == 'a':
            s = s+1

        if q5 == 'a':
            e = e+1

        if q5 == 'a':
            c = c+1


        q6 = request.POST['GROUP6']
        ans = Personality(questions = 6, answers = q6)
        ans.save()
        if q6 == 'c':
            r = r + 1

        if q6 == 'a':
            i = i+1

        if q6 == 'a':
            a = a+1

        if q6 == 'a':
            s = s+1

        if q6 == 'a':
            e = e+1

        if q6 == 'a':
            c = c+1


        q7 = request.POST['GROUP7']
        ans = Personality(questions = 7, answers = q7)
        ans.save()
        if q7 == 'c':
            r = r + 1

        if q7 == 'a':
            i = i+1

        if q7 == 'a':
            a = a+1

        if q7 == 'a':
            s = s+1

        if q7 == 'a':
            e = e+1

        if q7 == 'a':
            c = c+1


        q8 = request.POST['GROUP8']
        ans = Personality(questions = 8, answers = q8)
        ans.save()
        if q8 == 'a':
            r = r + 1

        if q8 == 'a':
            i = i+1

        if q8 == 'a':
            a = a+1

        if q8 == 'a':
            s = s+1

        if q8 == 'a':
            e = e+1

        if q8 == 'a':
            c = c+1


        q9 = request.POST['GROUP9']
        ans = Personality(questions = 9, answers = q9)
        ans.save()
        if q9 == 'd':
            r = r + 1

        if q9 == 'e':
            i = i+1

        if q9 == 'd':
            a = a+1

        if q9 == 'd':
            s = s+1

        if q9 == 'd':
            e = e+1

        if q9 == 'c':
            c = c+1


        q10 = request.POST['GROUP10']
        ans = Personality(questions = 10, answers = q10)
        ans.save()
        if q10 == 'd':
            r = r + 1

        if q10 == 'd':
            i = i+1

        if q10 == 'd':
            a = a+1

        if q10 == 'd':
            s = s+1

        if q10 == 'd':
            e = e+1

        if q10 == 'd':
            c = c+1


        q11 = request.POST['GROUP11']
        ans = Personality(questions = 11, answers = q11)
        ans.save()
        if q11 == 'd':
            r = r + 1

        if q11 == 'd':
            i = i+1

        if q11 == 'd':
            a = a+1

        if q11 == 'd':
            s = s+1

        if q11 == 'a':
            e = e+1

        if q11 == 'a':
            c = c+1


        q12 = request.POST['GROUP12']
        ans = Personality(questions = 12, answers = q12)
        ans.save()
        if q12 == 'c':
            r = r + 1

        if q12 == 'd':
            i = i+1

        if q12 == 'a':
            a = a+1

        if q12 == 'a':
            s = s+1

        if q12 == 'a':
            e = e+1

        if q12 == 'c':
            c = c+1


        q13 = request.POST['GROUP13']
        ans = Personality(questions = 13, answers = q13)
        ans.save()
        if q13 == 'd':
            r = r + 1

        if q13 == 'd':
            i = i+1

        if q13 == 'a':
            a = a+1

        if q13 == 'a':
            s = s+1

        if q13 == 'a':
            e = e+1

        if q13 == 'a':
            c = c+1


        q14 = request.POST['GROUP14']
        ans = Personality(questions = 14, answers = q14)
        ans.save()
        if q14 == 'd':
            r = r + 1

        if q14 == 'd':
            i = i+1

        if q14 == 'd':
            a = a+1

        if q14 == 'a':
            s = s+1

        if q14 == 'a':
            e = e+1

        if q14 == 'a':
            c = c+1


        q15 = request.POST['GROUP15']
        ans = Personality(questions = 15, answers = q15)
        ans.save()
        if q15 == 'c':
            r = r + 1

        if q15 == 'd':
            i = i+1

        if q15 == 'a':
            a = a+1

        if q15 == 'a':
            s = s+1

        if q15 == 'a':
            e = e+1

        if q15 == 'a':
            c = c+1


        q16 = request.POST['GROUP16']
        ans = Personality(questions = 16, answers = q16)
        ans.save()
        if q16 == 'a':
            r = r + 1

        if q16 == 'a':
            i = i+1

        if q16 == 'a':
            a = a+1

        if q16 == 'a':
            s = s+1

        if q16 == 'a':
            e = e+1

        if q16 == 'a':
            c = c+1


        q17 = request.POST['GROUP17']
        ans = Personality(questions = 17, answers = q17)
        ans.save()
        if q17 == 'a':
            r = r + 1

        if q17 == 'a':
            i = i+1

        if q17 == 'a':
            a = a+1

        if q17 == 'a':
            s = s+1

        if q17 == 'a':
            e = e+1

        if q17 == 'a':
            c = c+1


        q18 = request.POST['GROUP18']
        ans = Personality(questions = 18, answers = q18)
        ans.save()
        if q18 == 'a':
            r = r + 1

        if q18 == 'a':
            i = i+1

        if q18 == 'd':
            a = a+1

        if q18 == 'd':
            s = s+1

        if q18 == 'a':
            e = e+1

        if q18 == 'a':
            c = c+1


        q19 = request.POST['GROUP19']
        ans = Personality(questions = 19, answers = q19)
        ans.save()
        if q19 == 'a':
            r = r + 1

        if q19 == 'd':
            i = i+1

        if q19 == 'e':
            a = a+1

        if q19 == 'd':
            s = s+1

        if q19 == 'd':
            e = e+1

        if q19 == 'a':
            c = c+1


        q20 = request.POST['GROUP20']
        ans = Personality(questions = 20, answers = q20)
        ans.save()
        if q20 == 'a':
            r = r + 1

        if q20 == 'a':
            i = i+1

        if q20 == 'e':
            a = a+1

        if q20 == 'd':
            s = s+1

        if q20 == 'a':
            e = e+1

        if q20 == 'a':
            c = c+1


        q21 = request.POST['GROUP21']
        ans = Personality(questions = 21, answers = q21)
        ans.save()
        if q21 == 'a':
            r = r + 1
 
        if q21 == 'd':
            i = i+1

        if q21 == 'e':
            a = a+1

        if q21 == 'e':
            s = s+1

        if q21 == 'a':
            e = e+1

        if q21 == 'a':
            c = c+1


        q22 = request.POST['GROUP22']
        ans = Personality(questions = 22, answers = q22)
        ans.save()
        if q22 == 'e':
            r = r + 1

        if q22 == 'd':
            i = i+1

        if q22 == 'e':
            a = a+1

        if q22 == 'e':
            s = s+1

        if q22 == 'd':
            e = e+1

        if q22 == 'd':
            c = c+1


        q23 = request.POST['GROUP23']
        ans = Personality(questions = 23, answers = q23)
        ans.save()
        if q23 == 'a':
            r = r + 1

        if q23 == 'a':
            i = i+1

        if q23 == 'a':
            a = a+1

        if q23 == 'a':
            s = s+1

        if q23 == 'a':
            e = e+1

        if q23 == 'a':
            c = c+1


        q24 = request.POST['GROUP24']
        ans = Personality(questions = 24, answers = q24)
        ans.save()
        if q24 == 'a':
            r = r + 1

        if q24 == 'a':
            i = i+1

        if q24 == 'e':
            a = a+1

        if q24 == 'd':
            s = s+1

        if q24 == 'a':
            e = e+1

        if q24 == 'a':
            c = c+1


        q25 = request.POST['GROUP25']
        ans = Personality(questions = 25, answers = q25)
        ans.save()
        if q25 == 'd':
            r = r + 1

        if q25 == 'd':
            i = i+1

        if q25 == 'd':
            a = a+1

        if q25 == 'e':
            s = s+1

        if q25 == 'd':
            e = e+1

        if q25 == 'd':
            c = c+1


        q26 = request.POST['GROUP26']
        ans = Personality(questions = 26, answers = q26)
        ans.save()
        if q26 == 'd':
            r = r + 1

        if q26 == 'e':
            i = i+1

        if q26 == 'd':
            a = a+1

        if q26 == 'e':
            s = s+1

        if q26 == 'd':
            e = e+1

        if q26 == 'd':
            c = c+1


        q27 = request.POST['GROUP27']
        ans = Personality(questions = 27, answers = q27)
        ans.save()
        if q27 == 'c':
            r = r + 1
    
        if q27 == 'e':
            i = i+1

        if q27 == 'd':
            a = a+1

        if q27 == 'd':
            s = s+1

        if q27 == 'd':
            e = e+1

        if q27 == 'c':
            c = c+1


        q28 = request.POST['GROUP28']
        ans = Personality(questions = 28, answers = q28)
        ans.save()
        if q28 == 'd':
            r = r + 1

        if q28 == 'd':
            i = i+1

        if q28 == 'd':
            a = a+1

        if q28 == 'd':
            s = s+1

        if q28 == 'd':
            e = e+1

        if q28 == 'c':
            c = c+1


        q29 = request.POST['GROUP29']
        ans = Personality(questions = 29, answers = q29)
        ans.save()
        if q29 == 'd':
            r = r + 1
 
        if q29 == 'e':
            i = i+1

        if q29 == 'd':
            a = a+1

        if q29 == 'e':
            s = s+1

        if q29 == 'd':
            e = e+1

        if q29 == 'd':
            c = c+1


        q30 = request.POST['GROUP30']
        ans = Personality(questions = 30, answers = q30)
        ans.save()
        if q30 == 'd':
            r = r + 1

        if q30 == 'd':
            i = i+1

        if q30 == 'd':
            a = a+1

        if q30 == 'e':
            s = s+1

        if q30 == 'd':
            e = e+1

        if q30 == 'c':
            c = c+1


        q31 = request.POST['GROUP31']
        ans = Personality(questions = 31, answers = q31)
        ans.save()
        if q31 == 'c':
            r = r + 1

        if q31 == 'd':
            i = i+1

        if q31 == 'c':
            a = a+1

        if q31 == 'e':
            s = s+1

        if q31 == 'd':
            e = e+1

        if q31 == 'd':
            c = c+1


        q32 = request.POST['GROUP32']
        ans = Personality(questions = 32, answers = q32)
        ans.save()
        if q32 == 'c':
            r = r + 1

        if q32 == 'c':
            i = i+1

        if q32 == 'c':
            a = a+1

        if q32 == 'd':
            s = s+1

        if q32 == 'c':
            e = e+1

        if q32 == 'c':
            c = c+1


        q33 = request.POST['GROUP33']
        ans = Personality(questions = 33, answers = q33)
        ans.save()
        if q33 == 'a':
            r = r + 1

        if q33 == 'a':
            i = i+1

        if q33 == 'a':
            a = a+1

        if q33 == 'a':
            s = s+1

        if q33 == 'c':
            e = e+1

        if q33 == 'c':
            c = c+1


        q34 = request.POST['GROUP34']
        ans = Personality(questions = 34, answers = q34)
        ans.save()
        if q34 == 'a':
            r = r + 1

        if q34 == 'a':
            i = i+1

        if q34 == 'a':
            a = a+1

        if q34 == 'a':
            s = s+1

        if q34 == 'c':
            e = e+1

        if q34 == 'c':
            c = c+1


        q35 = request.POST['GROUP35']
        ans = Personality(questions = 35, answers = q35)
        ans.save()
        if q35 == 'c':
            r = r + 1

        if q35 == 'c':
            i = i+1

        if q35 == 'a':
            a = a+1

        if q35 == 'c':
            s = s+1

        if q35 == 'd':
            e = e+1

        if q35 == 'd':
            c = c+1


        q36 = request.POST['GROUP36']
        ans = Personality(questions = 36, answers = q36)
        ans.save()
        if q36 == 'a':
            r = r + 1

        if q36 == 'a':
            i = i+1

        if q36 == 'a':
            a = a+1

        if q36 == 'a':
            s = s+1

        if q36 == 'a':
            e = e+1

        if q36 == 'a':
            c = c+1


        q37 = request.POST['GROUP37']
        ans = Personality(questions = 37, answers = q37)
        ans.save()
        if q37 == 'd':
            r = r + 1

        if q37 == 'c':
            i = i+1

        if q37 == 'a':
            a = a+1

        if q37 == 'c':
            s = s+1

        if q37 == 'e':
            e = e+1

        if q37 == 'd':
            c = c+1


        q38 = request.POST['GROUP38']
        ans = Personality(questions = 38, answers = q38)
        ans.save()
        if q38 == 'a':
            r = r + 1

        if q38 == 'a':
            i = i+1

        if q38 == 'a':
            a = a+1

        if q38 == 'a':
            s = s+1

        if q38 == 'd':
            e = e+1

        if q38 == 'c':
            c = c+1


        q39 = request.POST['GROUP39']
        ans = Personality(questions = 39, answers = q39)
        ans.save()
        if q39 == 'a':
            r = r + 1

        if q39 == 'a':
            i = i+1

        if q39 == 'a':
            a = a+1

        if q39 == 'a':
            s = s+1

        if q39 == 'd':
            e = e+1

        if q39 == 'c':
            c = c+1

        q40 = request.POST['GROUP40']
        ans = Personality(questions = 40, answers = q40)
        ans.save()
        if q40 == 'c':
            r = r + 1

        if q40 == 'a':
            i = i+1

        if q40 == 'a':
            a = a+1

        if q40 == 'a':
            s = s+1

        if q40 == 'c':
            e = e+1

        if q40 == 'd':
            c = c+1


        q41 = request.POST['GROUP41']
        ans = Personality(questions = 41, answers = q41)
        ans.save()
        if q41 == 'a':
            r = r + 1

        if q41 == 'a':
            i = i+1

        if q41 == 'a':
            a = a+1

        if q41 == 'a':
            s = s+1

        if q41 == 'c':
            e = e+1

        if q41 == 'd':
            c = c+1


        q42 = request.POST['GROUP42']
        ans = Personality(questions = 42, answers = q42)
        ans.save()
        if q42 == 'c':
            r = r + 1

        if q42 == 'a':
            i = i+1

        if q42 == 'a':
            a = a+1

        if q42 == 'a':
            s = s+1

        if q42 == 'c':
            e = e+1

        if q42 == 'c':
            c = c+1


        q43 = request.POST['GROUP43']
        ans = Personality(questions = 43, answers = q43)
        ans.save()
        if q43 == 'c':
            r = r + 1

        if q43 == 'a':
            i = i+1

        if q43 == 'a':
            a = a+1

        if q43 == 'a':
            s = s+1

        if q43 == 'c':
            e = e+1

        if q43 == 'd':
            c = c+1


        q44 = request.POST['GROUP44']
        ans = Personality(questions = 44, answers = q44)
        ans.save()
        if q44 == 'c':
            r = r + 1

        if q44 == 'a':
            i = i+1

        if q44 == 'a':
            a = a+1

        if q44 == 'a':
            s = s+1

        if q44 == 'c':
            e = e+1

        if q44 == 'd':
            c = c+1


        q45 = request.POST['GROUP45']
        ans = Personality(questions = 45, answers = q45)
        ans.save()
        if q45 == 'd':
            r = r + 1

        if q45 == 'a':
            i = i+1

        if q45 == 'a':
            a = a+1

        if q45 == 'a':
            s = s+1

        if q45 == 'd':
            e = e+1

        if q45 == 'd':
            c = c+1


        q46 = request.POST['GROUP46']
        ans = Personality(questions = 46, answers = q46)
        ans.save()
        if q46 == 'c':
            r = r + 1
 
        if q46 == 'c':
            i = i+1

        if q46 == 'a':
            a = a+1

        if q46 == 'a':
            s = s+1

        if q46 == 'c':
            e = e+1

        if q46 == 'd':
            c = c+1


        q47 = request.POST['GROUP47']
        ans = Personality(questions = 47, answers = q47)
        ans.save()
        if q47 == 'a':
            r = r + 1

        if q47 == 'a':
            i = i+1

        if q47 == 'a':
            a = a+1

        if q47 == 'a':
            s = s+1

        if q47 == 'c':
            e = e+1

        if q47 == 'd':
            c = c+1


        q48 = request.POST['GROUP48']
        ans = Personality(questions = 48, answers = q48)
        ans.save()
        if q48 == 'c':
            r = r + 1

        if q48 == 'a':
            i = i+1

        if q48 == 'a':
            a = a+1

        if q48 == 'a':
            s = s+1

        if q48 == 'c':
            e = e+1

        if q48 == 'c':
            c = c+1


        maximum = max(r, i, a, s, e, c)

        if maximum == r:
            message = "Realistic"        
            message1 = "Engineers, Architecture, Dietitian, Criminal Justice, Forestry and Athelete"

        elif maximum == i:
            message = "Investigative"
            message1 = "Biology, Chemistry, Computer Science, Economics, Law, Mathematics, Psychology and Data Analyst"
        
        elif maximum == a:
            message = "Artistic"
            message1 = "Advertising, Art, Graphic Design, Music, Theater and Writer"

        elif maximum == s:
            message = "Social"
            message1 = "Communications, Anthropology, Education, Nursing, Religion, Sociology and Social Work "

        elif maximum == e:
            message = "Enterprising"
            message1 = "Business, Finance, Law Enforcement, Real Estate, Marketing and Journalism"

        elif maximum == c:
            message = "Conventional"
            message1 = "Accounting, Computer Information System, Library, Science and Administrative Work"       

        

        return render(request, 'resultp.html', {'message': message, 'message1': message1})
        
    else:
        return render (request,'mcq4.html')