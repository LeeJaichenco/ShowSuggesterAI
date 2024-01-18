import logging
import subprocess
from PIL import Image
from io import BytesIO
import requests
from fuzzywuzzy import process
import pandas as pd
import openai
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - Line: %(lineno)d  - %(levelname)s - %(message)s')
openai.api_key = os.getenv('OPENAI_API_KEY')


# Function to read TV show titles from a CSV file
def read_tv_shows_from_csv(csv_path='imdb_tvshows.csv'):
    df = pd.read_csv(csv_path)
    tv_shows = df['Title'].str.strip().tolist()
    return tv_shows


# Function to get user input
def get_user_input():
    while True:
        shows = input(
            "Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: ")
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


# Function to get the final user shows
def run_show_suggestion(csv_path):
    valid_shows = read_tv_shows_from_csv(csv_path)
    while True:
        user_shows = get_user_input()
        corrected_shows = check_shows(user_shows, valid_shows)
        print(f"Just to make sure, do you mean {', '.join(corrected_shows)}?")
        if input("(y/n): ").lower() == 'y':
            return corrected_shows
        else:
            print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.")


# Function to generate embeddings
def generate_embeddings(text):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


# Function to process the CSV file
def process_csv_to_dict(csv_path='imdb_tvshows.csv', pickle_path='tv_show_embeddings.pkl'):
    tv_show_embeddings = get_embeddings_dict_from_pickle(pickle_path)
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        title = row['Title']
        if title not in tv_show_embeddings:
            tv_show_data = f"{row['Title']} {row['Description']} {row['Genres']} {row['Actors']}"
            embedding = generate_embeddings(tv_show_data)
            tv_show_embeddings[title] = embedding
            logging.info(index)

    with open(pickle_path, 'wb') as handle:
        pickle.dump(tv_show_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tv_show_embeddings


# Function to get the embeddings from the pickle file
def get_embeddings_dict_from_pickle(pickle_path='tv_show_embeddings.pkl'):
    # Check if the pickle file exists
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as file:
            # Load and return the embeddings dictionary
            embeddings_dict = pickle.load(file)
            return embeddings_dict
    else:
        return {}


def calculate_average_vector(list_of_vectors):
    df = pd.DataFrame(list_of_vectors)
    # Calculate the mean of each column
    average_vector = df.mean(axis=0).tolist()
    return average_vector


def calculate_cosine_similarity(vec_a, vec_b):
    return cosine_similarity([vec_a], [vec_b])[0][0]


def recommend_shows(input_shows, embeddings_dict, average_vector):
    print("Great! Generating recommendations...")
    similarity_scores = {}

    for show, vector in embeddings_dict.items():
        if show not in input_shows:
            similarity = calculate_cosine_similarity(vector, average_vector)
            similarity_scores[show] = similarity
    # Sort the shows by similarity from high to low
    sorted_shows = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_shows_with_percentage = get_top_shows_with_percentage(sorted_shows)

    return top_shows_with_percentage


def get_top_shows_with_percentage(sorted_shows):
    old_min = min(sorted_shows, key=lambda x: x[1])[1]
    old_max = max(sorted_shows, key=lambda x: x[1])[1]
    top_shows = sorted_shows[:5]
    # Rescale similarity scores to percentage
    top_shows_with_percentage = [(show, rescale_similarity(score, old_min, old_max, 0, 100)) for show, score in
                                 top_shows]
    return top_shows_with_percentage


def rescale_similarity(similarity, old_min, old_max, new_min, new_max):
    return ((similarity - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


# Function that returns the recommendations as a string
def format_recommendations(recommended_shows):
    if not recommended_shows:
        return "No recommendations available."

    recommendation_str = "Here are the TV shows that I think you would love:\n"
    for show, percentage in recommended_shows:
        recommendation_str += f"{show} ({percentage:.0f}%)\n"

    return recommendation_str


# Function to generate and display invented shows and ads
def made_up_shows(recommended_shows, user_shows):
    description_prompt = f"Create a TV show description based on the shows below, in your answer provide the structure: show name- description "
    image_prompt = "Create an image ad for this show:."

    user_shows_description = generate_tv_show_description(description_prompt + str(user_shows))
    recommended_shows_description = generate_tv_show_description(description_prompt + (', '.join([show for (show, _) in recommended_shows])))

    print("I have also created just for you two shows which I think you would love. \n"
          "Show #1 is based on the fact that you loved the input shows that you"
          f" gave me, {user_shows_description}. \n\n"
          f"Show #2 is based on the shows that I recommended for you, {recommended_shows_description}. "
          "Here are also the 2 tv show ads. Hope you like them!")

    generate_tv_show_image(image_prompt + user_shows_description, "input_show.png")
    generate_tv_show_image(image_prompt + recommended_shows_description, "recommended_show.png")


def generate_tv_show_description(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        api_key=os.getenv('OPENAI_API_KEY')
    )
    # Access the generated content from the choices array
    generated_text = response['choices'][0]['message']['content']

    return generated_text.strip()


# Generate and display image using DALL-E API
def generate_tv_show_image(prompt, show_name):
    image_path = dall_e_generate_image(prompt, show_name)
    display_image(image_path)


def display_image(image_path):
    if os.name == 'posix':  # for Linux, Mac, etc.
        subprocess.run(['open', image_path])
    elif os.name == 'nt':  # for Windows
        os.startfile(image_path)


def dall_e_generate_image(prompt, output_dir='generated_images', image_name=None):
    if image_name is None:
        image_name = "generated_image.png"
    else:
        image_name = image_name

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, image_name)

    # Generate an image using DALL-E
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        api_key=os.getenv('OPENAI_API_KEY')
    )

    image_url = response['data'][0]['url']
    # Download and save the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image.save(output_path)

    return output_path


def run_tv_show_recommender_system():
    csv_path = 'imdb_tvshows.csv'
    pickle_path = 'tv_show_embeddings.pkl'
    embeddings_dict = process_csv_to_dict(csv_path, pickle_path)
    # Get users confirmed shows
    user_shows = run_show_suggestion(csv_path)
    # Generate embeddings for user's shows
    user_show_embeddings = [embeddings_dict[show] for show in user_shows if show in embeddings_dict]
    # Calculate the average vector of all the users shows
    average_vector = calculate_average_vector(user_show_embeddings)
    # Generate and display recommendations
    recommended_shows = recommend_shows(user_shows, embeddings_dict, average_vector)
    formatted_recommendations = format_recommendations(recommended_shows)
    print(formatted_recommendations)
    # Generate and display invented shows and ads
    made_up_shows(recommended_shows, user_shows)


if __name__ == "__main__":
    run_tv_show_recommender_system()
