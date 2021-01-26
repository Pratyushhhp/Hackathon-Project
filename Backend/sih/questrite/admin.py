from django.contrib import admin
from .models import Personality
from .models import School
from .models import Graduates
from .models import Dropouts

# Register your models here.

admin.site.register(Personality)
admin.site.register(School)
admin.site.register(Graduates)
admin.site.register(Dropouts)



