import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load datasets
gym_data = pd.read_csv("D:/ml/fitness_recommender/gym_members_exercise_tracking.csv")
scaled_calories = pd.read_csv("D:/ml/fitness_recommender/scaled_calories.csv")

# Ensure necessary columns exist
def ensure_column(df, column, default_value):
    if column not in df.columns:
        df[column] = default_value
    return df

gym_data = ensure_column(gym_data, 'goal', 'general fitness')
gym_data = ensure_column(gym_data, 'Gender', 'Male')

# Clean and preprocess data
gym_data.dropna(subset=['Age', 'Height (m)', 'Weight (kg)', 'goal'], inplace=True)

# Define input (X) and target (y)
X = gym_data[['Age', 'Height (m)', 'Weight (kg)', 'Gender']]
y = gym_data['goal']

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

# Evaluate model
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))

# Streamlit UI
st.set_page_config(page_title="AI Workout Recommender", layout="wide")
st.title("🔥 AI-Powered Workout Recommender 🏋")

# Sidebar for user input
st.sidebar.header("Enter User Data")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["fat loss", "muscle gain", "cut", "bulk", "general fitness"])

if st.sidebar.button("Get Workout Recommendation"):
    # Prepare user input
    user_input = pd.DataFrame([[age, height/100, weight, gender]], columns=["Age", "Height (m)", "Weight (kg)", "Gender"])
    user_input = pd.get_dummies(user_input, columns=["Gender"], drop_first=True)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    # Scale user input
    user_input[['Age', 'Height (m)', 'Weight (kg)']] = scaler.transform(user_input[['Age', 'Height (m)', 'Weight (kg)']])

    # Predict goal
    predicted_goal = knn.predict(user_input)[0]
    st.success(f"🏆 Recommended Goal: {goal}")

    # Suggested exercises based on goal
    exercise_plan = {
        "fat loss": ["🏃‍♂️ Treadmill - 30 min", "⏳ Jump Rope - 5 sets x 3 min", "🔥 Burpees - 3 sets x 15 reps", "🧘 Yoga Cooldown - 10 min"],
        "muscle gain": ["🏋️ Bench Press - 4 sets x 8 reps", "💪 Deadlift - 4 sets x 6 reps", "🏋️ Squats - 4 sets x 10 reps", "🌀 Foam Rolling - 10 min"],
        "cut": ["🚣 Rowing Machine - 20 min", "💥 Push-Ups - 3 sets x 20 reps", "🦵 Lunges - 3 sets x 15 reps each leg", "🧘 Stretching - 10 min"],
        "bulk": ["🏋️ Overhead Press - 4 sets x 6 reps", "🤸 Pull-Ups - 4 sets x 8 reps", "🦿 Leg Press - 4 sets x 10 reps", "💆 Deep Tissue Massage - 10 min"],
        "general fitness": ["🧍 Plank - 3 sets x 1 min", "🏋️ Kettlebell Swings - 3 sets x 15 reps", "🤸 Bodyweight Squats - 3 sets x 20 reps", "💪 Mobility Drills - 10 min"]
    }

    st.write("### 📝 Recommended Workout Plan")
    for exercise in exercise_plan.get(goal, ["❌ No specific plan found — please check your input."]):
        st.write(f"- {exercise}")

    # Motivational quotes
    motivation_quotes = [
        "🌟 Push yourself because no one else is going to do it for you!",
        "🏆 Success starts with self-discipline.",
        "💥 The body achieves what the mind believes.",
        "🔥 Wake up. Work out. Look hot. Kick ass. Repeat.",
        "✨ Don’t wish for it — work for it!"
    ]
    st.write(f"💪 Motivation: {np.random.choice(motivation_quotes)}")

    # Calories burned progress chart - smaller and clearer
    st.write("📊 Weekly Calorie Burn Progress")
    progress_df = pd.DataFrame({"Day": [f"Day {i}" for i in range(1, 8)], "Calories": np.random.randint(200, 600, 7)})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(progress_df["Day"], progress_df["Calories"], color='skyblue')
    ax.set_ylabel("Calories Burned")
    ax.set_xlabel("Day")
    ax.set_title("Calories Burned Over the Week")
    plt.xticks(rotation=45)
    st.pyplot(fig)
