# Peer Tutor Microservice

## Project Overview

The **Peer Tutor Microservice** is an ML-powered FastAPI microservice that identifies the most suitable peer tutors for students needing help in a specific academic topic. It uses a trained **Gradient Boosting Classifier** model, enhanced with Snorkel-based weak supervision and domain-specific heuristics, to recommend tutors based on match likelihood, tutor karma, and recency of past help.

---

## Setup Instructions

1. **Clone the Repository**

    ```
    git clone https://github.com/your-username/peer-tutor-matcher.git
    cd peer-tutor-matcher
    ```
2. **Create Virtual Environment**

    ```
    python3 -m venv venv
    ```
    For Mac users:
    ```
    source venv/bin/activate
    ```
    For Windows users:
    ```
    venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Run the App**
    ```
    uvicorn main:app --reload
    ```
    Open your web browser and navigate to `http://localhost:8000/docs` to access the API documentation. Or if you are offline you can use the /offline-docs endpoint.

---
## API Examples
 - **POST /match-peer-tutors**

 Request
 ```
 {
    "user_id": "stu_0000",
    "topic": "DBMS",
    "urgency_level": "low"
}
```

Response
```
{
    "user_id": "stu_0000",
    "matched_peers": [
        {
            "peer_id": "stu_1001",
            "name": "Alice A",
            "college": "Test College",
            "karma_in_topic": 95,
            "match_score": 0.9991,
            "predicted_help_probability": 0.9991,
            "last_helped_on": "2025-05-31",
            "match_reason": [
                "same college",
                "same branch",
                "same year",
                "recently active ( less than 7 days )",
                "high karma ( greater than 80 )"
            ]
        },
        {
            "peer_id": "stu_1002",
            "name": "Bob B",
            "college": "Test College",
            "karma_in_topic": 80,
            "match_score": 0.9991,
            "predicted_help_probability": 0.9991,
            "last_helped_on": "2025-05-30",
            "match_reason": [
                "same college",
                "same branch",
                "same year",
                "recently active ( less than 7 days )"
            ]
        },
        {
            "peer_id": "stu_2019",
            "name": "Student 19",
            "college": "Test College",
            "karma_in_topic": 79,
            "match_score": 0.9991,
            "predicted_help_probability": 0.9991,
            "last_helped_on": "2025-05-29",
            "match_reason": [
                "same college",
                "same branch",
                "same year"
            ]
        }
    ],
    "status": "success"
}
```
---
- **GET /health**

Response
```
{
    "status": "ok"
}
```
---
- **GET /version**

Response
```
{
    "model_version": "1.0.0"
}
```

## Test Instructions
```
python3 -m pytest -v
```
This will run the test_match_peers.py inside the tests/ folder

## Docker Instructions

Build the Docker image
```
docker build -t peer-tutor-matcher .
```
Run the Docker container
```
docker run -p 8000:8000 peer-tutor-matcher
```
This will start the server on port 8000

Run test in docker
```
docker run --rm -it peer-tutor-matcher python3 -m pytest -v
```







