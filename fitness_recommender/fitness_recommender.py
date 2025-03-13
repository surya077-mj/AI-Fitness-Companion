import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Embedded dataset (extracted from CSV files)
data = {
    "Age": [56, 46, 32, 25, 38],
    "Height (m)": [1.71, 1.53, 1.66, 1.70, 1.79],
    "Weight (kg)": [88.3, 74.9, 68.1, 53.2, 46.1],
    "Gender": ["Male", "Female", "Female", "Male", "Male"],
    "goal": ["general fitness", "fat loss", "muscle gain", "cut", "bulk"]
}
gym_data = pd.DataFrame(data)

# Preprocessing
gym_data = pd.get_dummies(gym_data, columns=["Gender"], drop_first=True)
scaler = StandardScaler()
gym_data[['Age', 'Height (m)', 'Weight (kg)']] = scaler.fit_transform(
    gym_data[['Age', 'Height (m)', 'Weight (kg)']]
)

# Splitting data
X = gym_data.drop(columns=["goal"])
y = gym_data["goal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Model
grid_search = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

def predict_goal(age, height, weight, gender):
    user_input = pd.DataFrame([[age, height / 100, weight, gender]], 
                              columns=["Age", "Height (m)", "Weight (kg)", "Gender"])
    user_input = pd.get_dummies(user_input, columns=["Gender"], drop_first=True)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)
    user_input[['Age', 'Height (m)', 'Weight (kg)']] = scaler.transform(user_input[['Age', 'Height (m)', 'Weight (kg)']])
    return knn.predict(user_input)[0]

# Streamlit UI
st.set_page_config(page_title="AI Workout Recommender", layout="wide")
st.title("🔥 AI-Powered Workout Recommender 🏋")

st.sidebar.header("Enter User Data")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

if st.sidebar.button("Get Workout Recommendation"):
    predicted_goal = predict_goal(age, height, weight, gender)
    st.success(f"🏆 Recommended Goal: {predicted_goal}")

    workout_plans = {
        "fat loss": ["Treadmill - 30 min", "Jump Rope - 5 sets x 3 min"],
        "muscle gain": ["Bench Press - 4 sets x 8 reps", "Deadlift - 4 sets x 6 reps"],
        "cut": ["Rowing Machine - 20 min", "Push-Ups - 3 sets x 20 reps"],
        "bulk": ["Overhead Press - 4 sets x 6 reps", "Pull-Ups - 4 sets x 8 reps"],
        "general fitness": ["Plank - 3 sets x 1 min", "Kettlebell Swings - 3 sets x 15 reps"]
    }

    st.write("### 📝 Recommended Workout Plan")
    for exercise in workout_plans.get(predicted_goal, ["No plan available"]):
        st.write(f"- {exercise}")

    st.write("📊 Weekly Calorie Burn Progress")
    progress_df = pd.DataFrame({"Day": [f"Day {i}" for i in range(1, 8)], "Calories": np.random.randint(200, 600, 7)})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(progress_df["Day"], progress_df["Calories"], color='skyblue')
    ax.set_ylabel("Calories Burned")
    ax.set_xlabel("Day")
    ax.set_title("Your Weekly Calorie Burn Progress")
    plt.xticks(rotation=45)
    st.pyplot(fig)
