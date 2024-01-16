import logging
from fuzzywuzzy import process
import csv
import pandas as pd
import openai
import os
import pickle


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - Line: %(lineno)d  - %(levelname)s - %(message)s')


# Function to read TV show titles from a CSV file
def read_tv_shows_from_csv(csv_path):
    tv_shows = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tv_shows.append(row['Title'].strip())
    return tv_shows


# Function to get user input
def get_user_input():
    while True:
        shows = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: ")
        user_shows = [show.strip() for show in shows.split(',')]
        if len(user_shows) > 1:
            return user_shows
        else:
            print("Please enter more than one TV show.")


# Function to check and correct user input using fuzzy string matching
def check_shows(user_shows, valid_shows):
    corrected_shows = []
    for user_show in user_shows:
        # Using 'extractOne' to find the closest match in valid_shows
        closest_match = process.extractOne(user_show, valid_shows)
        corrected_shows.append(closest_match[0])  # closest_match is a tuple (matched_string, score)
    return corrected_shows


# Function for recommending shows (placeholder)
def recommend_shows(shows):
    print("Great! Generating recommendations...")


def run_show_suggestion(csv_path):
    valid_shows = read_tv_shows_from_csv(csv_path)
    while True:
        user_shows = get_user_input()
        corrected_shows = check_shows(user_shows, valid_shows)
        print(f"Just to make sure, do you mean {', '.join(corrected_shows)}?")
        if input("(y/n): ").lower() == 'y':
            recommend_shows(corrected_shows)
            break
        else:
            print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.")


# Function to generate embeddings
def generate_embeddings(text):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    logging.info(openai.api_key)
    response = openai.Embedding.create(
        input=text, 
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


# Function to process the CSV file
def process_csv_to_dict(csv_path):
    if not os.path.exists('tv_show_embeddings.pkl'):
        # Read the CSV file
        df = pd.read_csv(csv_path)
        tv_show_embeddings = {}

        for index, row in df.head(1).iterrows():
            tv_show_data = f"{row['Title']} {row['Description']} {row['Genres']} {row['Actors']}"
            embedding = generate_embeddings(tv_show_data)
            tv_show_embeddings[row['Title']] = embedding

        # Save the dictionary using pickle
        with open('tv_show_embeddings.pkl', 'wb') as handle:
            pickle.dump(tv_show_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return tv_show_embeddings

    logging.info('The pickle file already exists.')


def calculate_average_vector(list_of_vectors):
    # Initialize a list with zeros of the same length as the first vector
    average_vector = [0] * len(list_of_vectors[0])

    # Sum up all vectors
    for vector in list_of_vectors:
        for i, value in enumerate(vector):
            average_vector[i] += value

    # Divide by the number of vectors to get the average
    num_vectors = len(list_of_vectors)
    average_vector = [x / num_vectors for x in average_vector]

    return average_vector

    
if __name__ == "__main__":
    file_path = 'imdb_tvshows.csv'
    process_csv_to_dict(file_path)
