import statsmodels.api as sm
from statsmodels.formula.api import ols


def perform_anova(results_df):
    # Convert hyperparameters to appropriate types
    results_df['num_heads'] = results_df['num_heads'].astype(str)
    results_df['out_channels'] = results_df['out_channels'].astype(str)
    
    # Define the formula for the ANOVA model
    formula = 'val_f1 ~ C(num_heads) + C(out_channels) + learning_rate + weight_decay'
    
    # Fit the model using Ordinary Least Squares (OLS)
    model = ols(formula, data=results_df).fit()
    
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table
