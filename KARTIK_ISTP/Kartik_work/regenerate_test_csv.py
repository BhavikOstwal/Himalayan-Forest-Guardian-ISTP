"""
Regenerate test_processed.csv after balancing test set
"""
from pathlib import Path
import pandas as pd

# Get current test files
test_chainsaw_dir = Path('data/processed/test/chainsaw')
test_non_chainsaw_dir = Path('data/processed/test/non_chainsaw')

# Collect file paths and labels
data = []

# Chainsaw files
for wav_file in test_chainsaw_dir.glob('*.wav'):
    data.append({
        'file_path': str(wav_file).replace('\\', '/'),
        'label': 1,
        'label_name': 'chainsaw'
    })

# Non-chainsaw files
for wav_file in test_non_chainsaw_dir.glob('*.wav'):
    data.append({
        'file_path': str(wav_file).replace('\\', '/'),
        'label': 0,
        'label_name': 'non_chainsaw'
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = 'data/processed/test_processed.csv'
df.to_csv(output_file, index=False)

print(f"Regenerated {output_file}")
print(f"Total samples: {len(df)}")
print(f"Chainsaw: {len(df[df['label']==1])}")
print(f"Non-chainsaw: {len(df[df['label']==0])}")
print(f"Ratio: {len(df[df['label']==1])*100/len(df):.1f}% chainsaw")
