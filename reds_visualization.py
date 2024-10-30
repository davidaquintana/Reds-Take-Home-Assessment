import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions_output data
file_path = "C:\\Users\\david\\OneDrive\\Desktop\\Misc\\Code\\Sports  Analytics\\Reds\\predictions_output.csv"
predictions_output = pd.read_csv(file_path)

# Visualize the distribution of pitch mix proportions across all batters with a boxplot
plt.figure(figsize=(10, 6))
predictions_output[['Proportion_FB', 'Proportion_BB', 'Proportion_OS']].plot.box()
plt.title("Distribution of Pitch Mix Proportions Across All Batters")
plt.ylabel("Proportion")
plt.xlabel("Pitch Type")
plt.show()

# Select a sample of batters to show individual pitch mix proportions
sample_batters = predictions_output.sample(5, random_state=42)  # Randomly selecting 5 batters for example

# Plot a stacked bar chart for pitch mix proportions of sample batters
sample_batters.plot(
    x='PLAYER_NAME', 
    y=['Proportion_FB', 'Proportion_BB', 'Proportion_OS'], 
    kind='bar', stacked=True, figsize=(10, 6)
)
plt.title("Pitch Mix Proportions for Sample Batters")
plt.ylabel("Proportion")
plt.xlabel("Player Name")
plt.legend(title="Pitch Type")
plt.show()
