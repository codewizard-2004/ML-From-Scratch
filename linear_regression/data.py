import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Using pandas for easier CSV saving with column names

# --- 1. Set a random seed (for reproducibility) ---
np.random.seed(42)

# --- 2. Define the True Underlying Relationship (the "perfect" line) ---
# Let's imagine: Score = 3 * Study_Time + 40 (meaning for every hour studied, score increases by 3, with a baseline of 40)
true_slope = 3       # 'm' - increase in score per hour of study
true_intercept = 40  # 'b' - baseline score (if study time is 0)

# --- 3. Generate Study Time (Independent Variable X) ---
# Let's simulate study times from 0 to 10 hours
num_students = 100 # How many data points (students)
study_time = np.linspace(0, 10, num_students) # Study times from 0 to 10 hours

# --- 4. Generate Scores (Dependent Variable Y) based on relationship and add noise ---
# Calculate the 'perfect' score without any randomness
perfect_score = true_slope * study_time + true_intercept

# Add some random noise to simulate real-world variability (e.g., some students are better test-takers, some days are off)
# The noise will be centered around 0. Let's make the spread (standard deviation) reasonable.
noise_strength = 7 # Adjust this to control how much scatter you see in scores
noise = np.random.normal(loc=0, scale=noise_strength, size=num_students)

# Combine the perfect scores with the noise to get our final scores
score = perfect_score + noise

# --- 5. (Optional) Visualize the generated data ---
plt.figure(figsize=(10, 7))
plt.scatter(study_time, score, label='Student Data Points', alpha=0.7, color='green')
plt.plot(study_time, perfect_score, color='red', linestyle='--', linewidth=2, label='True Underlying Score Trend')

plt.title('Simulated Student Study Time vs. Exam Score')
plt.xlabel('Study Time (Hours)')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()

# --- 6. Combine data into a Pandas DataFrame for easy CSV saving with headers ---
# While numpy.savetxt works, pandas.DataFrame.to_csv is often more convenient for labeled data.
data = pd.DataFrame({
    'Study_Time_Hours': study_time,
    'Exam_Score': score
})

# --- 7. Save the data to CSV ---
csv_file_name = 'study_time_vs_score.csv'
data.to_csv(csv_file_name, index=False) # index=False prevents pandas from writing the DataFrame index as a column

print(f"\nFirst 5 rows of the generated data:")
print(data.head())
print(f"\nData saved successfully to {csv_file_name}")