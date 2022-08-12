from django.views.generic.base import View
from django.shortcuts import redirect,render
import pandas as pd
import joblib

class Index(View):
    template_name='IndexTemplates.html'
    def get(self,*args,**kwargs):
        context={
            'title':'Hallo Gaes',
            'prediksi':'predict'
        }
        return render(self.request,self.template_name,context)
    def post(self,*args,**kwargs):
        ###Pregnancies=self.request.POST['pregnancies'] pada POST['pregnancies'] merupakan name yang harus sama dengan name di HTML nya
        Pregnancies=self.request.POST['pregnancies']
        Glucose=self.request.POST['glocose']
        BloodPressure=self.request.POST['bloodpressure']
        SkinThickness=self.request.POST['skinthickness']
        Insulin=self.request.POST['insulin']
        BMI=self.request.POST['bmi']
        DiabetesPedigreeFunction=self.request.POST['diabetespedigreefunction']
        Age=self.request.POST['age']

        clf = joblib.load("./data_model/model.pkl")## ini merupakan pemanggilan modelnya
        X = pd.DataFrame(
            [[Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]], #ini samakan dengan atribut di codingan pembuatan modelnya samakan dengan yang didataset
            columns = ["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]) #ini sama juga samakan dengan atribut di codingan pembuatan modelnya samakan dengan yang didataset

        prediction = clf.predict(X)[0]
        if prediction == 1:
            prediction="Anda Kena Diabetes"
        else:
            prediction="Anda Normal, Jangan banyak makan gula"
        context={
            'prediksi':prediction
        }
        return render(self.request,self.template_name,context)