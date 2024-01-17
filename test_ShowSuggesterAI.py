import ShowSuggesterAI as SSA
import pytest
import math


@pytest.fixture
def all_shows():
    return SSA.read_tv_shows_from_csv("imdb_tvshows.csv")


@pytest.fixture
def example_shows():
    return ["Game of Thrones", "Breaking Bad"]


@pytest.fixture
def average_vector(example_shows):
    embeddings_dict = SSA.process_csv_to_dict("imdb_tvshows.csv")  # Get embeddings dictionary
    show_vectors = [embeddings_dict[show] for show in example_shows if
                    show in embeddings_dict]  # Get vectors for example shows
    return SSA.calculate_average_vector(show_vectors)


@pytest.fixture
def recommended_shows(example_shows, average_vector):
    embeddings_dict = SSA.process_csv_to_dict("imdb_tvshows.csv")
    return SSA.recommend_shows(example_shows, embeddings_dict, average_vector)


def test_read_tv_shows_from_csv(example_shows, all_shows):
    # Only check the first 2 items for a match
    assert all_shows[:2] == example_shows


def test_check_shows_corrects_input(example_shows):
    user_shows = ["Game of Thre", "Brking Bed"]
    valid_shows = ["Game of Thrones", "Breaking Bad", "The Witcher"]
    assert SSA.check_shows(user_shows, valid_shows) == example_shows


def test_process_csv_to_dict_contains_all_shows(all_shows):
    # Process the CSV to get the embeddings dictionary
    embeddings_dict = SSA.process_csv_to_dict("imdb_tvshows.csv")

    # Check that every show name is a key in the embeddings dictionary
    for show_name in all_shows:
        assert show_name in embeddings_dict, f"Embeddings missing for show: {show_name}"


def test_calculate_average_vector():
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example embedding vectors
    expected_average = [4, 5, 6]  # Expected average vector
    assert SSA.calculate_average_vector(vectors) == expected_average


def test_cosine_similarity():
    vector_a = [1, 0, -1]
    vector_b = [-1, 0, 1]
    expected_similarity = -1  # Cosine similarity for opposite vectors
    actual_similarity = SSA.calculate_cosine_similarity(vector_a, vector_b)
    assert math.isclose(actual_similarity, expected_similarity, rel_tol=1e-9)


# Test for the correct number of recommended shows
def test_recommend_shows_length(recommended_shows):
    assert len(recommended_shows) == 5


# Test that recommended shows do not include input shows
def test_recommend_shows_excludes_input(example_shows, recommended_shows):
    recommended_show_names = [show[0] for show in recommended_shows]  # Assuming recommended_shows returns a list of tuples or list of names
    assert all(show not in example_shows for show in recommended_show_names)


def test_dall_e_generate_image_returns_valid_path():
    prompt = "Test prompt for DALL-E"
    filename = "test_image.png"

    image_path = SSA.dall_e_generate_image(prompt, image_name=filename)

    # Check if the return value is a valid image path
    assert image_path.endswith(filename)
