import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load datasets from GitHub
GITHUB_RAW_URL = "https://raw.githubusercontent.com/surya077-mj/AI-Fitness-Companion/main/fitness_recommender/"
gym_data_url = GITHUB_RAW_URL + "gym_members_exercise_tracking.csv"
scaled_calories_url = GITHUB_RAW_URL + "scaled_calories.csv"

def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"🚨 Error loading {url}: {e}")
        return None

gym_data = load_data(gym_data_url)
scaled_calories = load_data(scaled_calories_url)

# Ensure dataset is loaded
if gym_data is None or scaled_calories is None:
    st.stop()

# Ensure necessary columns exist
def ensure_column(df, column, default_value):
    if column not in df.columns:
        df[column] = default_value
    return df

gym_data = ensure_column(gym_data, 'Gender', 'Male')

# Clean and preprocess data
gym_data.dropna(subset=['Age', 'Height (m)', 'Weight (kg)'], inplace=True)

# Define input (X) and target (y)
X = gym_data[['Age', 'Height (m)', 'Weight (kg)', 'Gender']]
y = np.random.choice(["Cardio", "Strength Training", "HIIT", "Flexibility", "General Fitness"], len(gym_data))  # Dummy target

# Encode categorical variables
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X[['Age', 'Height (m)', 'Weight (kg)']] = scaler.fit_transform(X[['Age', 'Height (m)', 'Weight (kg)']])

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fine-tune KNN model
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_k = grid_search.best_params_['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="AI Workout Recommender", layout="wide")
st.title("🔥 AI-Powered Workout Recommender 🏋")

# Sidebar for user input
st.sidebar.header("Enter User Data")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
exercise_type = st.sidebar.selectbox("Exercise Type", ["Cardio", "Strength Training", "HIIT", "Flexibility", "General Fitness"])

if st.sidebar.button("Get Workout Recommendation"):
    st.success(f"🏆 Selected Exercise Type: {exercise_type}")

    # Suggested exercises based on exercise type
    exercise_plan = {
        "Cardio": ["🏃‍♂️ Treadmill - 30 min", "🚴 Cycling - 20 min", "⏳ Jump Rope - 5 sets x 3 min", "🧘 Yoga Cooldown - 10 min"],
        "Strength Training": ["🏋️ Bench Press - 4 sets x 8 reps", "💪 Deadlift - 4 sets x 6 reps", "🏋️ Squats - 4 sets x 10 reps", "🌀 Foam Rolling - 10 min"],
        "HIIT": ["🔥 Burpees - 3 sets x 15 reps", "💥 Jump Squats - 3 sets x 12 reps", "🚀 Mountain Climbers - 3 sets x 30 sec", "🧘 Stretching - 10 min"],
        "Flexibility": ["🧘 Yoga Flow - 20 min", "🦵 Dynamic Stretching - 10 min", "🤸 Mobility Drills - 15 min", "💆 Deep Tissue Massage - 10 min"],
        "General Fitness": ["🧍 Plank - 3 sets x 1 min", "🏋️ Kettlebell Swings - 3 sets x 15 reps", "🤸 Bodyweight Squats - 3 sets x 20 reps", "💪 Mobility Drills - 10 min"]
    }
    
    st.write("### 📝 Recommended Workout Plan")
    for exercise in exercise_plan.get(exercise_type, ["❌ No specific plan found."]):
        st.write(f"- {exercise}")
    
    # Motivational quote
    motivation_quotes = [
        "🌟 Push yourself because no one else is going to do it for you!",
        "🏆 Success starts with self-discipline.",
        "💥 The body achieves what the mind believes.",
        "🔥 Wake up. Work out. Look hot. Kick ass. Repeat.",
        "✨ Don’t wish for it — work for it!"
    ]
    st.write(f"💪 Motivation: {np.random.choice(motivation_quotes)}")
    
    # Calories burned progress chart
    st.write("📊 Weekly Calorie Burn Progress")
    progress_df = pd.DataFrame({"Day": [f"Day {i}" for i in range(1, 8)], "Calories": np.random.randint(200, 600, 7)})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(progress_df["Day"], progress_df["Calories"], color='skyblue')
    ax.set_ylabel("Calories Burned")
    ax.set_xlabel("Day")
    ax.set_title("Calories Burned Over the Week")
    plt.xticks(rotation=45)
    st.pyplot(fig)