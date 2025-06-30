import json
import random

random.seed(42)

MAX_INACTIVE_DAYS = 14

def generateDataset(sample_size=600):
    dataset = []
    for i in range(sample_size // 2):
        positive_sample = generatePositiveSample()
        dataset.append(positive_sample)

        negative_sample = generateNegativeSample()
        dataset.append(negative_sample)
    random.shuffle(dataset)
    return dataset

def generatePositiveSample():
    # Strong positive: high karma, recent help, rest can vary
    karma_in_topic = random.randint(70, 100)
    days_since_last_help = random.randint(0, 4)
    same_branch = random.choice([True, True, False])
    same_college = random.choice([True, False])
    peer_year_match = random.choice([True, False])

    # Slightly weaker positive: mid karma, but very recent help and matching branch/college
    if random.random() < 0.3:
        karma_in_topic = random.randint(60, 79)
        days_since_last_help = random.randint(0, 3)
        same_branch = True
        same_college = random.choice([True, False])

    # Compose label logic similar to snorkel combos
    positiveFactors = (
        2 * (karma_in_topic >= 80) +
        (60 <= karma_in_topic < 80) +
        (days_since_last_help <= 4) +
        same_branch +
        same_college +
        peer_year_match
    )
    label = 1 if positiveFactors >= 4 else 0

    return {
        'input': {
            'karma_in_topic': karma_in_topic,
            'same_college': same_college,
            'days_since_last_help': days_since_last_help,
            'same_branch': same_branch,
            'peer_year_match': peer_year_match
        },
        'label': label
    }

def generateNegativeSample():
    # Strong negative: low karma, inactive, rest can vary
    karma_in_topic = random.randint(0, 25)
    days_since_last_help = random.randint(MAX_INACTIVE_DAYS + 1, 30)
    same_branch = random.choice([False, False, True])
    same_college = random.choice([False, True])
    peer_year_match = random.choice([False, True])

    # Slightly weaker negative: mid-low karma, but inactive and mismatched branch/college
    if random.random() < 0.3:
        karma_in_topic = random.randint(26, 59)
        days_since_last_help = random.randint(10, 20)
        same_branch = False
        same_college = random.choice([False, True])

    negativeFactors = sum([
        days_since_last_help > MAX_INACTIVE_DAYS,
        karma_in_topic < 25,
        not same_branch,
        not same_college,
        not peer_year_match
    ])
    label = 0 if negativeFactors >= 3 else 1

    return {
        'input': {
            'karma_in_topic': karma_in_topic,
            'same_college': same_college,
            'days_since_last_help': days_since_last_help,
            'same_branch': same_branch,
            'peer_year_match': peer_year_match
        },
        'label': label
    }

def saveDataset(dataset, filename='datasetH.json'):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    dataset = generateDataset()
    saveDataset(dataset)
    print(f"Generated dataset with {len(dataset)} samples.")