from django.views.generic.base import View
from django.shortcuts import redirect,render
import pandas as pd
import joblib

class TitanicView(View):
    template_name='TitanicView.html'
    def get(self,*args,**kwargs):
        context={
            'title':'Dlobo | Data Science',
            'header':'Selamat datang di Aplikasi prediksi kemungkinan kehidupan pada kasus tenggelamnya kapal titanic'
        }
        return render(self.request,self.template_name,context)
    def post(self,*args,**kwargs):
        clf = joblib.load("./Models/model.pkl")
        SibSp=self.request.POST['jml_saudara']
        Parch=self.request.POST['jml_anak']
        Fare=self.request.POST['trf_tiket']
        Pclass=self.request.POST['kls_tiket'].split(',')
        Pclass[0]=int(Pclass[0])
        Pclass[1]=int(Pclass[1])
        Pclass[2]=int(Pclass[2])
        Sex=self.request.POST['jns_kelamin'].split(',')
        Sex[0]=int(Sex[0])
        Sex[1]=int(Sex[1])
        Embarked=self.request.POST['tpt_keberangkatan'].split(',')
        Embarked[0]=int(Embarked[0])
        Embarked[1]=int(Embarked[1])
        Embarked[2]=int(Embarked[2])
        

        X = pd.DataFrame(
            [[SibSp,
            Parch,
            Fare,
            Pclass[0],
            Pclass[1],
            Pclass[2],
            Sex[0],
            Sex[1],
            Embarked[0],
            Embarked[1],
            Embarked[2]
            ]],
            columns = [
                "SibSp", 
                "Parch",
                "Fare",
                "Pclass_1",
                "Pclass_2",
                "Pclass_3",
                "Sex_female",
                "Sex_male",
                "Embarked_C",
                "Embarked_Q",
                "Embarked_S"])
        prediction = clf.predict(X)[0]
        if prediction == 1:
            prediction="Anda memiliki kemungkinan selamat"
        else:
            prediction="Anda Normal, Jangan banyak makan gula"
        context={
            'prediksi':prediction
        }
        return render(self.request,self.template_name,context)