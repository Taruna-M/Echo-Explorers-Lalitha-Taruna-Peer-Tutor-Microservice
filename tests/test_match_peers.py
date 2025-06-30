import pytest
from fastapi.testclient import TestClient
from main import app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

# Constants
REQUESTER_ID = "stu_0000"
TOPIC = "DBMS"
URL = "/match-peer-tutors"

def post_request(urgency="medium"):
    payload = {
        "user_id": REQUESTER_ID,
        "topic": TOPIC,
        "urgency_level": urgency
    }
    return client.post(URL, json=payload)

def extract_peer_by_id(response_json, peer_id):
    return next((p for p in response_json["matched_peers"] if p["peer_id"] == peer_id), None)


# Functional Tests
# F1: Valid topic with active tutors
def test_valid_active_tutors_returned():
    response = post_request()
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert (len(data["matched_peers"]) <= 3 and len(data["matched_peers"]) > 0)  # At least one match, max 3

    for peer in data["matched_peers"]:
        assert peer["match_score"] >= 0.7 

# F2: Inactive tutor (stu_1004) is excluded
def test_inactive_peer_is_filtered_out():
    response = post_request()
    data = response.json()
    assert extract_peer_by_id(data, "stu_1004") is None  # Dan D, inactive for 25 days

# F3: Low-karma tutor (stu_1005) is excluded
def test_low_karma_peer_is_filtered_out():
    response = post_request()
    data = response.json()
    assert extract_peer_by_id(data, "stu_1005") is None  # Eve E, karma = 5

# F4: Branch-based score comparison - adjust students.json to ensure this works
def test_same_branch_peer_has_higher_score_than_different_branch():
    response = post_request()
    data = response.json()
    matched_peers = data["matched_peers"]

    # Separate peers into same-branch and different-branch
    same_branch_scores = []
    diff_branch_scores = []
    
    for peer in matched_peers:
        if "same branch" in peer.get("match_reason", []):
            same_branch_scores.append(peer["match_score"])
        else:
            diff_branch_scores.append(peer["match_score"])

    # Ensure both groups exist for the test to be meaningful
    assert len(same_branch_scores) > 0, "No same-branch peers matched"
    assert len(diff_branch_scores) > 0, "No different-branch peers matched"

    # Check average score difference
    avg_same = sum(same_branch_scores) / len(same_branch_scores)
    avg_diff = sum(diff_branch_scores) / len(diff_branch_scores)

    assert avg_same > avg_diff, f"Expected same-branch peers to have higher average score ({avg_same:.2f} vs {avg_diff:.2f})"

# Edge Cases
def send_match_request(user_id, topic, urgency="medium"):
    return client.post(URL, json={
        "user_id": user_id,
        "topic": topic,
        "urgency_level": urgency
    })


# E-1: Invalid user_id (not in dataset)
def test_invalid_user_id_returns_404():
    response = send_match_request("nonexistent_user", "DBMS", "medium")
    assert response.status_code == 404
    assert "not found" in response.json().get("detail", "").lower()


# E-2: Invalid urgency_level
def test_invalid_urgency_level_returns_422():
    response = send_match_request("stu_0000", "DBMS", "urgent")
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"].lower() == "validation error"
    assert any("urgency_level" in str(item["loc"]) and "input should be" in item["msg"].lower()
               for item in data.get("detail", []))


# E-3: Valid user but topic not in any peer's karma
def test_valid_user_but_topic_not_in_peers():
    response = send_match_request("stu_0000", "QuantumComputing", "medium")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["matched_peers"] == []  # empty match list

# E-4: No eligible peers (all filtered out due to inactivity or low score)
def test_no_eligible_peers_returns_empty_list():
    # Assuming topic "OldTopic" is present but all peers are inactive (>14 days)
    response = send_match_request("stu_0000", "OldTopic", "medium")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["matched_peers"] == []


# E-5: Only requesting student has the topic
def test_requester_is_only_one_with_topic():
    # Assuming "SoloTopic" is present only in stu_0000 (requester)
    response = send_match_request("stu_0000", "SoloTopic", "medium")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["matched_peers"] == []


# E-6: Missing topic field in payload (triggers 422 from Pydantic)
def test_missing_topic_returns_422():
    invalid_payload = {
        "user_id": "stu_0000",
        "urgency_level": "medium"
        # topic missing!
    }
    response = client.post(URL, json=invalid_payload)
    assert response.status_code == 422

    data = response.json()
    assert data["status"] == "error"
    assert data["message"].lower() == "validation error"
    assert any("topic" in str(item["loc"]) for item in data.get("detail", []))

# Performance Tests
# P-1: Large peer pool performance and correctness
def test_large_peer_pool_performance_and_correctness():
    payload = {
        "user_id": "stu_0000",
        "topic": "DBMS",
        "urgency_level": "high"
    }

    response = client.post("/match-peer-tutors", json=payload)
    assert response.status_code == 200

    data = response.json()
    matched = data["matched_peers"]

    # Only up to 5 results
    assert len(matched) <= 5, f"Expected ≤ 5 matches, got {len(matched)}"

    # All match_score ≥ threshold
    assert all(peer["match_score"] >= 0.7 for peer in matched), "Some matches below threshold"

    # Scores are sorted descending
    scores = [peer["match_score"] for peer in matched]
    assert scores == sorted(scores, reverse=True), "Scores not sorted correctly"

    # Duration is reasonable (arbitrary upper bound: 200ms)
    duration = data.get("duration", 0)
    assert duration < 10, f"Slow response: {duration:.2f} ms"


# P-2 threshold test - adjust students.json to ensure this works
def test_threshold_edge_behavior():
    payload = {
        "user_id": "stu_0000",
        "topic": "DBMS",
        "urgency_level": "medium"
    }

    response = client.post("/match-peer-tutors", json=payload)
    assert response.status_code == 200

    data = response.json()
    matched_ids = [p["peer_id"] for p in data["matched_peers"]]

    assert "stu_1004" not in matched_ids, "Low Score peer should NOT be matched"
    assert "stu_2019" in matched_ids, "Above Threshold peer SHOULD be matched"

    print("\nRT-03 Matches:")
    for peer in data["matched_peers"]:
        print(f"{peer['peer_id']} | {peer['match_score']}")

# P-3: Strong matches ranked above weak ones
def test_strong_matches_ranked_above_weak_ones():
    response = post_request()
    assert response.status_code == 200
    data = response.json()
    matches = data["matched_peers"]

    # Extract match scores
    scores = [peer["match_score"] for peer in matches]

    # Ensure scores are strictly descending or non-increasing
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
        f"Match scores not sorted descending: {scores}"

    # Sanity: top match should be one of the strong peers (stu_1001 or stu_1002)
    top_peer_ids = [peer["peer_id"] for peer in matches[:2]]
    assert any(pid in top_peer_ids for pid in ["stu_1001", "stu_1002"]), \
        f"Top matches not among expected strong peers: {top_peer_ids}"


