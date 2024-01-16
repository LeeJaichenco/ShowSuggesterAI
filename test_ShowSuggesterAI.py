import ShowSuggesterAI as SSA
import pytest


@pytest.fixture
def all_shows():
    return SSA.read_tv_shows_from_csv("imdb_tvshows.csv")


@pytest.fixture
def example_shows():
    return ["Game of Thrones", "Breaking Bad"]


@pytest.fixture
def recommended_shows(all_shows, example_shows):
    return SSA.recommend_shows(example_shows, all_shows)


def test_read_tv_shows_from_csv(example_shows, all_shows):
    # Only check the first 2 items for a match
    assert all_shows[:2] == example_shows


def test_check_shows_corrects_input(example_shows):
    user_shows = ["Game of Thre", "Brking Bed"]
    valid_shows = ["Game of Thrones", "Breaking Bad", "The Witcher"]
    assert SSA.check_shows(user_shows, valid_shows) == example_shows


def test_calculate_average_vector():
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example embedding vectors
    expected_average = [4, 5, 6]  # Expected average vector
    assert SSA.calculate_average_vector(vectors) == expected_average


def test_cosine_similarity():
    vector_a = [1, 0, -1]
    vector_b = [-1, 0, 1]
    expected_similarity = -1  # Cosine similarity for opposite vectors
    assert SSA.cosine_similarity(vector_a, vector_b) == expected_similarity


# Test for the correct number of recommended shows
def test_recommend_shows_length(recommended_shows):
    assert len(recommended_shows) == 5


# Test that recommended shows do not include input shows
def test_recommend_shows_excludes_input(example_shows, recommended_shows):
    assert all(show not in example_shows for show in recommended_shows)

