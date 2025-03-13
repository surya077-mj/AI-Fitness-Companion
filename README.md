# AI Fitness Recommender

## Live Demo
[Click here to try the AI Fitness Recommender](https://ai-fitness-companion-surya.streamlit.app/)

## Overview
The **AI Fitness Recommender** is a Streamlit-based application that helps users get personalized workout recommendations based on their age, height, weight, and gender. It utilizes a dataset of gym members and exercise tracking to suggest workout plans for different fitness goals.

## Features
- User-friendly Streamlit UI
- Input fields for Age, Height, Weight, and Gender
- Predefined workout plans based on fitness goals
- Motivational quotes to keep users inspired
- Weekly calorie burn progress visualization

## Dataset
The application loads the following datasets from GitHub:
1. **gym_members_exercise_tracking.csv**
2. **scaled_calories.csv**

## How It Works
1. Users enter their details in the sidebar.
2. Based on the provided data, a recommended workout plan is displayed.
3. Motivational quotes appear to keep users engaged.
4. A bar chart shows a simulated weekly calorie burn progress.

## Installation & Usage
### Prerequisites
- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- scikit-learn

### Steps to Run
1. Clone the repository or download the source code.
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
4. Open the provided local URL in a web browser.

## Future Improvements
- Integration with real-time fitness APIs
- More personalized workout recommendations
- User progress tracking and history

## License
This project is open-source and available for modification and redistribution.
