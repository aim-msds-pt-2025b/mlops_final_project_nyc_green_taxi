import json
import urllib.request

payload = {
    "trip_distance": 2.3,
    "passenger_count": 1,
    "PULocationID": 74,
    "DOLocationID": 166,
    "hour": 14,
    "day_of_week": 2,
    "payment_type": 1,
}
headers = {"Content-Type": "application/json"}
req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
)
print(urllib.request.urlopen(req).read().decode("utf-8"))
