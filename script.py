# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# Print the first five rows
print(codecademy.head())
# Create a scatter plot of score vs completed
plt.scatter(codecademy.completed, codecademy.score)
plt.title("Score vs. Lesson Completed")
plt.xlabel("Lessons Completed")
plt.ylabel("Score on Quiz")
# Show then clear plot
plt.show()
plt.clf()
# Fit a linear regression to predict score based on prior lessons completed
model = sm.OLS.from_formula('score ~ completed', data = codecademy)
results = model.fit()
print(results.params)
# Intercept interpretation:
# A learner who has completed 0 lessons may score 13.2 on a quiz
# Slope interpretation:
# A learner is expected to get a 1.3 point increase in score for every lesson completed
# Plot the scatter plot with the line on top
plt.scatter(codecademy.completed,codecademy.score)
plt.plot(codecademy.completed,results.predict(codecademy))
plt.title("Score vs. Lesson Completed")
plt.ylabel("Lessons Completed")
plt.xlabel("Score on Quiz")
# Show then clear plot
plt.show()
plt.clf()
# Predict score for learner who has completed 20 prior lessons
#line equation
print(1.3*20+13.2)
# Calculate fitted values
fitted_values = results.predict(codecademy)
# Calculate residuals
residuals = codecademy.score - fitted_values
# Check normality assumption
plt.hist(residuals)
plt.show()
# Show then clear the plot
plt.clf()
# Check homoscedasticity assumption
plt.scatter(fitted_values,residuals)
plt.title("Homoscedasticity")
plt.xlabel("fitted Values")
plt.ylabel("residuals")
# Show then clear the plot
plt.show()
plt.clf()
# Create a boxplot of score vs lesson
sns.boxplot(x='lesson',y='score', data = codecademy)
plt.title("score vs lesson")
# Show then clear plot
plt.show()
plt.clf()
# Fit a linear regression to predict score based on which lesson they took
model = sm.OLS.from_formula('score ~ lesson', data = codecademy)
results = model.fit()
print(results.params)
# Calculate and print the group means and mean difference (for comparison)
lesson_A = np.mean(codecademy.score[codecademy.lesson == 'Lesson A'])
lesson_B = np.mean(codecademy.score[codecademy.lesson == 'Lesson B'])
mean_difference = lesson_A - lesson_B
print("Lesson A: ", lesson_A)
print("Lesson B: ", lesson_B)
print("Mean Difference: ", mean_difference)
# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
sns.lmplot(x='completed', y='score', hue = 'lesson', data = codecademy)
plt.title("Score vs Completed")
plt.show()
plt.clf()