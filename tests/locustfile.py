"""
Locust Load Testing for Crop Yield Prediction API.

Run with: locust -f locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random


class CropYieldAPIUser(HttpUser):
    """Simulates API users for load testing."""
    
    wait_time = between(1, 3)
    
    @task(10)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health")
    
    @task(30)
    def predict_single(self):
        """Test single prediction endpoint."""
        crops = ["wheat", "rice", "corn", "soybean"]
        regions = ["north", "south", "east", "west", "central"]
        
        payload = {
            "crop_type": random.choice(crops),
            "region": random.choice(regions),
            "sowing_date": f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-2024",
            "ndvi": round(random.uniform(0.3, 0.9), 2),
            "precipitation_mm": round(random.uniform(400, 1500), 1),
            "temperature_c": round(random.uniform(10, 35), 1),
            "soil_organic_carbon_pct": round(random.uniform(0.5, 4.0), 1)
        }
        
        self.client.post("/predict", json=payload)
    
    @task(5)
    def predict_batch(self):
        """Test batch prediction endpoint."""
        crops = ["wheat", "rice", "corn", "soybean"]
        regions = ["north", "south", "east", "west", "central"]
        
        batch_size = random.randint(2, 10)
        predictions = []
        
        for _ in range(batch_size):
            predictions.append({
                "crop_type": random.choice(crops),
                "region": random.choice(regions),
                "sowing_date": f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-2024",
                "ndvi": round(random.uniform(0.3, 0.9), 2),
                "precipitation_mm": round(random.uniform(400, 1500), 1),
                "temperature_c": round(random.uniform(10, 35), 1),
                "soil_organic_carbon_pct": round(random.uniform(0.5, 4.0), 1)
            })
        
        self.client.post("/predict_batch", json={"predictions": predictions})
    
    @task(3)
    def get_crops(self):
        """Test crops endpoint."""
        self.client.get("/crops")
    
    @task(3)
    def get_regions(self):
        """Test regions endpoint."""
        self.client.get("/regions")
    
    @task(2)
    def model_info(self):
        """Test model info endpoint."""
        self.client.get("/model/info")
