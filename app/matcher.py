# Main script for the matching logic among the peers

# Libraries
import logging
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

# Custom imports
from app.config import load_config
from app.appError import AppError


# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Constants
DATA_PATH = Path("data/students.json")
GBC_MODEL_PATH = Path("model/model.pkl")
MODEL_THRESHOLD = config["model_threshold"]
MAX_RESULTS = config["max_results"]
MAX_INACTIVE_DAYS = config["max_inactive_days"]
SORT_CONSTANT = 1e-4

# Cache and indexes
_student_lookup = {}
_topic_index = defaultdict(list)
_college_index = defaultdict(set)
_branch_index = defaultdict(set)
_year_index = defaultdict(set)
_date_cache = {}
_current_date = datetime.strptime("2025-06-02", "%Y-%m-%d").date()
# _current_date = datetime.now().date()

def _parse_cached_date(date_str: str) -> datetime.date:
    """Parse date string and cache the result to avoid repeated parsing
    
    Parameters:
        date_str (str): Date string in the format "YYYY-MM-DD"

    Returns:
        datetime.date: Parsed date object
    """
    if date_str not in _date_cache:
        _date_cache[date_str] = datetime.strptime(date_str, "%Y-%m-%d").date()
    return _date_cache[date_str]

def _build_indexes(student_data: List[Dict]):
    """Build indexes for fast lookups

    Parameters:
        student_data (List[Dict]): List of student data dictionaries containing peer_id, name, college, branch, year, karma_in_topic, and last_helped_on
    """
    global _student_lookup, _topic_index, _college_index, _branch_index, _year_index

    _student_lookup.clear()
    _topic_index.clear()
    _college_index.clear()
    _branch_index.clear()
    _year_index.clear()

    for student in student_data:
        peer_id = student['peer_id']
        _student_lookup[peer_id] = student

        # Only students with karma in topic are indexed
        for topic in student['karma_in_topic'].keys():
            _topic_index[topic].append(peer_id)

        # Index by college, branch, and year
        _college_index[student['college']].add(peer_id)
        _branch_index[student['branch']].add(peer_id)
        _year_index[student['year']].add(peer_id)

def _active_status(student: Dict, topic: str) -> Tuple[bool, int]:
    """Check if student for the topic is active based on days since last help

    Parameters:
        student (Dict): Student data dictionary
        topic (str): Topic name

    Returns:
        Tuple[bool, int]: Tuple containing is_active (bool) and days since last help (int)
    """
    last_helped_on = student['last_helped_on'].get(topic)
    if not last_helped_on:
        return False, MAX_INACTIVE_DAYS + 1
    last_helped_date = _parse_cached_date(last_helped_on)
    days_since_last_help = (_current_date - last_helped_date).days
    is_active = days_since_last_help <= MAX_INACTIVE_DAYS
    return is_active, days_since_last_help

# Get l
def valid_peers(user_id: str, topic: str) -> List[Tuple[Dict, int]]:
    """Get list of valid peers using indexes and those with karma in specified topic
    
    Parameters:
        user_id (str): Peer ID of the requesting user
        topic (str): Topic for which to find valid peers
    
    Returns:
        List[Tuple[Dict, int]]: List of tuples containing student data and days since last help
    """
    peers = _topic_index.get(topic, [])
    if not peers:
        return []
    
    valid_peers = []
    for peer_id in peers:
        # Skip requesting user
        if peer_id == user_id:
            continue

        student = _student_lookup[peer_id]
        is_active, days_since_last_help = _active_status(student, topic)
        if is_active:
            valid_peers.append((student, days_since_last_help))
    return valid_peers

def extract_features(requesting_student: Dict, peer_data_list: List[Tuple[Dict, int]], topic: str) -> np.ndarray:
    """Extract features vectors for the valid peers for model prediction
    
    Parameters:
        requesting_student (Dict): Student data dictionary of the requesting student
        peer_data_list (List[Tuple[Dict, int]]): List of tuples containing student data and days since last help
        topic (str): Topic for which to extract features 
    
    Returns:
        np.ndarray: Numpy array of shape (n_peers, 5) containing features for each peer
    """
    peers_no = len(peer_data_list)

    # Allocate features array for the 5 feats
    features = np.zeros((peers_no, 5)) 

    # Requesting student features
    req_branch = requesting_student['branch']
    req_college = requesting_student['college']
    req_year = requesting_student['year']

    for i, (peer, days_since_last_help) in enumerate(peer_data_list):
        features[i] = [
            peer['karma_in_topic'].get(topic, 0),
            1 if peer['college'] == req_college else 0,
            days_since_last_help,
            1 if peer['branch'] == req_branch else 0,
            1 if peer['year'] == req_year else 0
        ]
    return features

# Matched peers based on model prediction
def create_matches(peer_data_list: List[Tuple[Dict, int]], probabilities: np.ndarray, features: np.ndarray, topic: str, threshold: float) -> List[Dict]:
    """Create matches based on model predictions and features
    
    Parameters:
        peer_data_list (List[Tuple[Dict, int]]): List of tuples containing student data and days since last help
        probabilities (np.ndarray): Numpy array of predicted probabilities from the model
        features (np.ndarray): Numpy array of features for each peer
        topic (str): Topic for which to create matches
        threshold (float): Adjusted model threshold ( based on urgency ) for filtering matches based on probability
    
    Returns:
        List[Dict]: List of dictionaries containing matched peer data with reasons and scores 
    """
    valid_matches = probabilities >= threshold
    if not np.any(valid_matches):
        return []
    
    matches = []
    valid_indices = np.where(valid_matches)[0]

    for index in valid_indices:
        peer, days_since_last_help = peer_data_list[index]
        probability = probabilities[index]
        karma = int(features[index, 0])

        reasons = []
        if features[index, 1] == 1: reasons.append('same college')
        if features[index, 3] == 1: reasons.append('same branch')
        if features[index, 4] == 1: reasons.append('same year')
        if days_since_last_help <= 3: reasons.append('recently active ( less than 7 days )')
        if karma > 80: reasons.append('high karma ( greater than 80 )')

        match = {
            'peer_id': peer['peer_id'],
            'name': peer['name'],
            'college': peer['college'],
            'karma_in_topic': karma,
            'match_score': float(probability),
            'predicted_help_probability': float(probability),
            'last_helped_on': peer['last_helped_on'].get(topic, 'N/A'),
            'days_since_last_help': days_since_last_help,
            'match_reason': reasons,
        }
        matches.append(match)
    
    return matches

# Group similar scores together for tie breaker
def tie_breaker(x, urgency_level: str) -> Tuple:
    """Tie breaker function to group similar scores together
    
    Parameters:
        x (Dict): Match dictionary
        urgency_level (str): Urgency level of the request

    Returns:
        Tuple: Tuple containing grouped score and features for sorting
    """
    grouped_score = round(x['match_score'] / SORT_CONSTANT) * SORT_CONSTANT
    if urgency_level == "high":
        return (grouped_score, x["karma_in_topic"], -x["days_since_last_help"])
    else:
        return (grouped_score, x["karma_in_topic"])

# Load student data and build indexes
try:
    with open(DATA_PATH, 'r') as f:
        student_data = json.load(f)
    _build_indexes(student_data)
    logger.info(f'Loaded {len(student_data)} students and built indexes')
except FileNotFoundError:
    logger.error(f'Student data file {DATA_PATH} not found')
    raise AppError('Student data file not found', 500)

# Load model
try:
    model = joblib.load(GBC_MODEL_PATH)
    logger.info(f'Model loaded from {GBC_MODEL_PATH}')
except FileNotFoundError:
    logger.error(f'GBC Model path {GBC_MODEL_PATH} not found')
    raise AppError('Model file not found', 500)

# Match function to find peers for a given student id, topic, urgency
def match_peers(user_id: str, topic: str, urgency_level: str) -> Dict:
    """Match peers for a given user based on topic and urgency level

    Parameters:
        user_id (str): Peer ID of the requesting user
        topic (str): Topic for which to find peers
        urgency_level (str): Urgency level of the request ('low', 'medium', 'high')
    
    Returns:
        Dict: Dictionary containing (str) user_id, (list) matched peers, and (str) status 
    """

    # For low urgency increase the threshold to be more selective
    adjusted_threshold = MODEL_THRESHOLD + 0.1 if urgency_level == 'low' else MODEL_THRESHOLD

    # Find the requesting student
    requesting_student = _student_lookup.get(user_id)
    if not requesting_student:
        logger.error(f'Student {user_id} not found')
        raise AppError('Student {user_id} not found', 404)
    
    # Get valid peers for the topic
    peer_data_list = valid_peers(user_id, topic)
    
    if not peer_data_list:
        logger.info(f'No valid peers found for user {user_id} on topic {topic}')
        return {
            'user_id': user_id,
            'matched_peers': [],
            'status': 'success'
        }
    
    try:
        # Extract features for the valid peers
        feature_arr = extract_features(requesting_student, peer_data_list, topic)

        # Numpy arr -> DataFrame for model prediction
        feature_cols = ['karma_in_topic', 'same_college', 'days_since_last_help', 'same_branch', 'peer_year_match']
        feature_df = pd.DataFrame(feature_arr, columns=feature_cols)

        # Model prediction of valid peers
        probabilities = model.predict_proba(feature_df)[:, 1]
        
        # Create matches based on model predictions
        matches = create_matches(peer_data_list, probabilities, feature_arr, topic, adjusted_threshold)

        # Sort matches by score and apply tie breaker
        matched_peers = sorted(matches, key= lambda x: tie_breaker(x, urgency_level), reverse=True)
        matched_peers = matched_peers[:MAX_RESULTS+2] if urgency_level == 'high' else matched_peers[:MAX_RESULTS]
        
        # Remove days_since_last_help from the final output and round predictions
        for match in matched_peers:
            match['predicted_help_probability'] = round(match['predicted_help_probability'], 4)
            match['match_score'] = round(match['match_score'], 4)
            match.pop('days_since_last_help', None)
        return {
            'user_id': user_id,
            'matched_peers': matched_peers,
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f'Error during matching: {str(e)}')
        raise AppError(f'Error during matching: {str(e)}', 500)

# If student data changes then refresh the indexes
def refresh_indexes():
    """Refresh the indexes for student data"""
    global student_data
    try:
        with open(DATA_PATH, "r") as f:
            student_data = json.load(f)
        _build_indexes(student_data)
        logger.info(f"Refreshed indexes for {len(student_data)} students")
    except FileNotFoundError as e:
        logger.error(f"Student Data not found at {DATA_PATH}: {str(e)}")
        raise AppError(f"Student Data not found", 500)
