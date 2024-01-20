# TV Show Suggester AI

TV Show Suggester AI is a sophisticated recommendation system that provides personalized TV show suggestions. Utilizing a combination of machine learning techniques and natural language processing, it offers users a curated list of shows based on their preferences.

## Features

- **User Input Interpretation**: Accepts user preferences for TV shows.
- **Fuzzy String Matching**: Corrects user input to ensure accurate interpretation of show titles.
- **Content-Based Recommendation**: Analyzes TV show data to find similar content to user preferences.
- **Embedding Generation**: Utilizes OpenAI's embedding model to understand and compare show descriptions.
- **Cosine Similarity Scoring**: Ranks recommendations based on similarity to user's preferences.
- **Creative Content Generation**: Produces invented TV show descriptions and generates promotional image ads using DALL-E.

## How It Works

1. **Input Your Favorites**: Start by entering your favorite TV shows. For instance, if you love "Breaking Bad", "Narcos", "Sherlock" and "The Crown", simply type in these titles.
   
   <img width="874" alt="image" src="https://github.com/LeeJaichenco/ShowSuggesterAI/assets/130974535/e56e6378-e73c-471a-82b5-ce498ff06ac7">


2. **AI Analysis**: The AI delves into a comprehensive database of TV shows, analyzing details from plots, genres, and character dynamics based on your preferences.
   
3. **Receive Curated Recommendations**: Get a list of five shows that resonate with your taste. For example, it might suggest "Broadchurch", "Downton Abbey", "The Tudors", "Elementary", and "The Wire".
   
   <img width="407" alt="image" src="https://github.com/LeeJaichenco/ShowSuggesterAI/assets/130974535/d2ae41d9-7407-4423-8472-4e4764eeffbe">


4. **Discover Invented Shows**: The AI creatively generates two new show concepts complete with compelling descriptions, tailored to your preferences.
   
   <img width="882" alt="image" src="https://github.com/LeeJaichenco/ShowSuggesterAI/assets/130974535/714708f3-ffda-49ba-860e-dab36e17f07c">


5. **Visual Ads Creation**: Experience these invented shows come to life with captivating visual ads, rendering the narratives in vivid details.
   
   <img width="400" alt="image" src="https://github.com/LeeJaichenco/ShowSuggesterAI/assets/130974535/b03e94ac-4b89-42ca-a593-d8d9edd884de">  <img width="400" alt="image" src="https://github.com/LeeJaichenco/ShowSuggesterAI/assets/130974535/957e7952-3f8a-4bac-afbb-99dd24fa2982">


   


Experience a new realm of storytelling with TV Show Suggester AI, where your next favorite show is just a recommendation away.

## Installation

Before you begin, ensure you have met the following requirements:

- Python 3.6+
- Pip for installing dependencies

To install TV Show Suggester AI, follow these steps:

```bash
git clone https://github.com/yourusername/tv-show-suggester-ai.git
cd tv-show-suggester-ai
pip install -r requirements.txt
```
## Usage
To use TV Show Suggester AI, follow these steps:

1. Make sure you have the **.env** file in your project directory with your **OPENAI_API_KEY** set.
2. Run the program:
```bash
python tv_show_suggester_ai.py
```
