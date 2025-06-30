import json
import random
from datetime import datetime, timedelta

# Define subjects mapped to branches
branch_subject_map = {
    "CSE": [
        "DBMS", "Algorithms", "Data Structures", "Operating Systems",
        "Networks", "Software Engineering", "Machine Learning",
        "Artificial Intelligence", "Computer Architecture"
    ],
    "ECE": [
        "Digital Electronics", "Signals and Systems", "Microprocessors",
        "Communication Systems", "Control Systems", "Networks",
        "Embedded Systems"
    ],
    "IT": [
        "DBMS", "Software Engineering", "Networks",
        "Web Technologies", "Cloud Computing", "Cyber Security"
    ],
    "Mechanical": [
        "Thermodynamics", "Fluid Mechanics", "Machine Design",
        "Heat Transfer", "Manufacturing Processes"
    ],
    "Civil": [
        "Structural Analysis", "Geotechnical Engineering",
        "Transportation Engineering", "Environmental Engineering",
        "Hydraulics"
    ]
}

colleges = ["CVR College of Engineering", "VNR Vignana Jyothi", "IIIT Hyderabad"]
branches = list(branch_subject_map.keys())
first_names = ["Priya", "Sushanth", "Amit", "Neha", "Rajesh", "Sneha", "Karthik", "Divya", "Anil", "Meena"]
last_names = ["S", "M", "K", "R", "P", "T", "N", "L", "G", "D"]

num_students = 20

def random_date_within_days(days=16):
    return (datetime.now() - timedelta(days=random.randint(0, days))).strftime("%Y-%m-%d")

def generate_student(i):
    peer_id = f"stu_{1000 + i}"
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    college = random.choice(colleges)
    branch = random.choice(branches)
    year = random.randint(1, 4)
    
    # Start with branch-related subjects
    main_subjects = set(branch_subject_map[branch])
    
    # Get subjects from other branches
    other_branches = [b for b in branches if b != branch]
    other_subjects = set()
    for b in other_branches:
        other_subjects.update(branch_subject_map[b])
    
    # Select 0 to 2 random subjects from other branches to add
    cross_branch_subjects = random.sample(list(other_subjects), k=random.randint(0, 2))
    
    # Combine and pick 3 to 6 subjects total
    possible_subjects = list(main_subjects.union(cross_branch_subjects))
    subject_count = min(random.randint(3, 6), len(possible_subjects))
    chosen_subjects = random.sample(possible_subjects, subject_count)
    
    karma = {}
    last_helped = {}

    for subj in chosen_subjects:
        # Set karma to be higher (60-100) for main branch subjects, else 30-80
        if subj in main_subjects:
            karma[subj] = random.randint(60, 100)
        else:
            karma[subj] = random.randint(30, 80)
        
        last_helped[subj] = random_date_within_days(14)
    
    student = {
        "peer_id": peer_id,
        "name": name,
        "college": college,
        "branch": branch,
        "year": year,
        "karma_in_topic": karma,
        "last_helped_on": last_helped
    }
    return student

students = [generate_student(i) for i in range(num_students)]

with open("students.json", "w") as f:
    json.dump(students, f, indent=2)

print(f"Generated {num_students} students with mostly branch-specific subjects plus some cross-branch subjects.")
