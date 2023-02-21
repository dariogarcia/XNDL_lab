import pandas as pd
import statsmodels.api as sm
import numpy as np

data = pd.read_csv('../data/OnlineNewsPopularity.csv')
# Create the Poisson regression model
model = sm.GLM(data[' shares'], data[[' n_tokens_title',' title_sentiment_polarity',' num_hrefs',' num_imgs',' num_videos',' is_weekend']], family=sm.families.Poisson())

# Fit the model and print the summary
result = model.fit()
print(result.summary())

# Calculate the incidence rate ratios and 95% confidence intervals for each predictor variable
irrs = result.params.apply(lambda x: round(np.exp(x), 2))
lower_ci = result.conf_int()[0].apply(lambda x: round(np.exp(x), 2))
upper_ci = result.conf_int()[1].apply(lambda x: round(np.exp(x), 2))
print('Incidence Rate Ratios:\n', irrs)
print('95% Confidence Intervals:\n', pd.concat([lower_ci, upper_ci], axis=1))
