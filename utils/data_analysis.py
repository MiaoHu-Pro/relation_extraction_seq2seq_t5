import json
import statistics
from pathlib import Path

# Load the data from the JSON lines file
file_path = Path('../data/nyt/train.json')
data = [json.loads(line) for line in file_path.read_text().splitlines()]

# Calculate number of samples
num_samples = len(data)

# Calculate text lengths in characters
text_lengths = [len(entry['text']) for entry in data]
mean_text_length = statistics.mean(text_lengths)

# Calculate length of spo_list for each entry and find the maximum
spo_lengths = [len(entry['spo_list']) for entry in data]
max_spo_length = max(spo_lengths)
mean_spo_length = statistics.mean(spo_lengths)

# Display the results
print(f"Number of samples: {num_samples}")
print(f"Mean length of 'text' (in characters): {mean_text_length:.2f}")
print(f"Maximum length of 'spo_list': {max_spo_length}")
print(f"mean length of 'spo_list': {mean_spo_length}")

