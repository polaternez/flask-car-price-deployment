import requests

url = "http://localhost:5000/predict_api"
r = requests.post(url, json={"year":2016, "mileage":4000, "tax":325, "mpg":30.1, "engineSize":4.0})

print(r.json())