# Data Dictionary

| Feature | Type | Description | Expected Values |
|---|---|---|---|
| trip_distance | float | Trip distance in miles | >= 0 |
| passenger_count | int | Number of passengers | 0..6 |
| PULocationID | int | Pickup zone ID | 1..max |
| DOLocationID | int | Dropoff zone ID | 1..max |
| payment_type | int | Payment code (1=Credit, etc.) | 0..6 |
| hour | int | Pickup hour of day | 0..23 |
| day_of_week | int | Pickup day of week | 0..6 |
| duration_min | float | **Target** — trip duration in minutes | 1..120 (filtered) |
