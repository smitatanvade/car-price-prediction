from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pkl

app = Flask(__name__)

ds = pd.read_csv("cleaned_data.csv")
lra = pkl.load(open("LRA.pkl", "rb"))

@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/car-project")
def car_project():
    companies = sorted(ds["company"].unique())
    names = sorted(ds["name"].unique())
    fuel_types = sorted(ds["fuel_type"].unique())
    return render_template("car-project.html", companies = companies, names = names, fuel_types = fuel_types)


@app.route("/car-project-result")
def car_project_result():
    company = request.args.get("company")
    name = request.args.get("name")
    year = request.args.get("year")
    kms_driven = request.args.get("kms_driven")
    fuel_type = request.args.get("fuel_type")

    mydata = [company, name, year, kms_driven, fuel_type]
    myinput = pd.DataFrame(columns = ['company', 'name', 'year', 'kms_driven', 'fuel_type'], data = np.array(mydata).reshape(1, 5))
    result = round(lra.predict(myinput)[0,0], 2)

    return render_template("car-project-result.html", company = company, name = name, year = year, kms_driven = kms_driven, fuel_type = fuel_type, result = result)
   