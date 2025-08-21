# Dataset Notes

- **Source:** NYC TLC Green Taxi Trip Records (Jan 2024), parquet via CloudFront.
- **Target:** Trip duration in **minutes**, computed from pickup & dropoff timestamps.
- **Rows:** Thousands (satisfies 500+ minimum).
- **Suitability for Drift:** We simulate drift by perturbing key feature distributions (distance, hour, passenger_count) and by adding noise to the target.
