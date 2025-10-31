
import pandas as pd
import re

print("Starting data cleaning process...")

# Load the dataset
df = pd.read_csv('twcs.csv')
print(f"Dataset loaded with {len(df)} rows.")

# --- Step 1: Separate customer tweets and agent replies ---

# Get all tweets that are replies to another tweet
replies_df = df[pd.notna(df['in_response_to_tweet_id'])].copy()

# Get all original tweets (not replies)
# We assume the first tweet from a user to a company is the start of an inquiry
first_inquiry_tweets = df[df['inbound'] == True]
first_inquiry_ids = first_inquiry_tweets.groupby('author_id')['created_at'].idxmin()
inquiries_df = df.loc[first_inquiry_ids]

# --- Step 2: Match inquiries with their first replies ---

# Merge the two dataframes to find the first reply to each inquiry
merged_df = pd.merge(
    inquiries_df,
    replies_df,
    left_on='tweet_id',
    right_on='in_response_to_tweet_id',
    suffixes=('_inquiry', '_reply')
)

print(f"Found {len(merged_df)} matched inquiry-reply pairs.")

# --- Step 3: Clean the text and create prompt/completion columns ---

def clean_text(text):
    """Removes @mentions and URLs to clean up the text."""
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = text.strip()
    return text

merged_df['prompt'] = merged_df['text_inquiry'].apply(clean_text)
merged_df['completion'] = merged_df['text_reply'].apply(clean_text)


# --- Step 4: Filter out low-quality pairs and save ---

# Select only the final columns we need
final_df = merged_df[['prompt', 'completion']]

# Remove any rows where prompt or completion is empty after cleaning
final_df = final_df[final_df['prompt'] != '']
final_df = final_df[final_df['completion'] != '']

# Remove very short conversations (less than 20 characters)
final_df = final_df[final_df['prompt'].str.len() > 20]
final_df = final_df[final_df['completion'].str.len() > 20]

print(f"Filtered down to {len(final_df)} high-quality pairs.")

# Save the clean data to a new CSV file
output_filename = 'training_data.csv'
final_df.to_csv(output_filename, index=False)

print(f"Success! Clean data saved to '{output_filename}'.")