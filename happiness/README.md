# Analysis Report

## Data Analysis and Insights
### Story Narration from Life Satisfaction Dataset

#### 1. What the Data Reveals

The dataset comprises information on different countries, detailing various factors influencing life satisfaction, measured through the "Life Ladder" score. It covers a total of 2,363 entries across 165 unique countries, with the data predominantly spanning the years 2005 to 2023. Key metrics include Log GDP per capita, social support, healthy life expectancy at birth, freedom to make life choices, generosity, perceptions of corruption, and levels of positive and negative affect.

#### 2. The Analysis Performed

The analysis involved several statistical and visual techniques:

- **Descriptive Statistics**: Summary statistics provided insights into the mean, standard deviation, minimum, maximum, and quartiles for various factors associated with life satisfaction.
- **Correlation Heatmap**: A visual representation showcased how different variables relate to one another, highlighting strong correlations and associations.
- **PCA (Principal Component Analysis) Scatter Plot**: Used to reduce dimensionality and visualize the main components that capture the most variance in the dataset.
- **KMeans Clustering**: This technique categorized countries into groups based on similarities in the data, revealing patterns in life satisfaction based on underlying factors.
- **Outliers Plot**: Identified countries or cases that deviate significantly from the norm, warranting further investigation.

#### 3. Key Findings and Implications

- **Mean Life Satisfaction**: The average Life Ladder score was found to be 5.48, suggesting a moderate level of life satisfaction among the global population.
- **Economic Stability vs. Life Satisfaction**: A significant correlation was unveiled between Log GDP per capita and Life Ladder scores, indicating that wealthier nations tend to report higher satisfaction levels.
- **Social Support**: A strong relationship emerged between social support and life satisfaction (mean of 0.81), emphasizing the importance of community and connections in enhancing personal well-being.
- **Healthy Life Expectancy**: Countries with higher healthy life expectancy (mean of 63.4 years) also tended to score higher on the life satisfaction scale, reinforcing the ties between physical health and overall happiness.
- **Corruption Perception**: There was a notable inverse relationship between perceptions of corruption and life satisfaction, suggesting that countries with lower corruption levels likely foster a sense of trust and well-being among their populace.
- **Emotional Affect**: The dataset revealed that countries with higher positive affect scores (mean of 0.651) also exhibited higher life satisfaction, while negative affects had a modest correlation, indicating a complex interplay between the emotional states of individuals and their reported life satisfaction.

#### 4. Conclusion

This analysis paints a comprehensive picture of factors impacting life satisfaction across different nations. The dataset fundamentally illustrates that economic prosperity, social connectivity, health, and corruption perceptions significantly shape how citizens perceive their lives. 

The study's implications suggest that policymakers aiming to enhance life satisfaction should focus on improving social support systems, fostering economic growth, addressing corruption, and promoting health initiatives. Given the nuanced relationships identified, future research could delve deeper into cultural and regional variations that affect these dynamics, further illuminating the path towards improving global well-being. 

In summary, the narrative built from this data not only contributes to our understanding of happiness in a global context but also emphasizes our interconnectedness and the shared human pursuit of a fulfilling life.

### Generated Visualizations
- [Correlation Heatmap](correlation_matrix.png)
- [PCA Scatter Plot](pca_scatter.png)
- [KMeans Clustering](kmeans_clustering.png)
- [Outliers Plot](outliers.png)
