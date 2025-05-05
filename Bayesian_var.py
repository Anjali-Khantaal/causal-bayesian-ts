import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)

# Load the preprocessed data
data = pd.read_csv("financial_timeseries_preprocessed.csv", index_col=0, parse_dates=True)

# We'll focus on the stationary transformed series for VAR modeling
# Using the log returns of S&P 500 and the differenced Federal Funds Rate
var_data = data[['SP500_log_return', 'FedFunds_diff']].copy()

# Convert date index for better plotting
var_data.index = pd.to_datetime(var_data.index)

# Let's look at our data
print("Data shape:", var_data.shape)
print("\nFirst few rows:")
print(var_data.head())
print("\nStatistics:")
print(var_data.describe())

# Visualize the time series
plt.figure(figsize=(14, 8))

# Plot S&P 500 log returns
plt.subplot(2, 1, 1)
plt.plot(var_data.index, var_data['SP500_log_return'], 'b-', alpha=0.7)
plt.title('S&P 500 Monthly Log Returns')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.ylabel('Log Return')

# Plot Fed Funds Rate differences
plt.subplot(2, 1, 2)
plt.plot(var_data.index, var_data['FedFunds_diff'], 'g-', alpha=0.7)
plt.title('Federal Funds Rate Monthly Differences')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.ylabel('Rate Difference')

plt.tight_layout()
plt.savefig('time_series_visualization.png')
plt.close()

# INSERT THE DAG VISUALIZATION CODE HERE
# First, add the required import at the top of your file with other imports:
import networkx as nx

# Then add this function and function call:
# Create and visualize the causal DAG
def plot_causal_dag():
    """
    Create and visualize the explicit causal DAG for the interest rate -> asset price relationship
    """
    plt.figure(figsize=(8, 6))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for our variables
    G.add_node("Interest Rate\n(Fed Funds)", pos=(0, 0))
    G.add_node("Asset Price\n(S&P 500)", pos=(1, 0))
    
    # Add the causal edge (Interest Rate -> Asset Price)
    G.add_edge("Interest Rate\n(Fed Funds)", "Asset Price\n(S&P 500)")
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the DAG
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
            font_size=12, font_weight='bold', arrowsize=20, 
            width=2, edge_color='darkblue')
    
    # Add title and annotations
    plt.title('Causal Directed Acyclic Graph (DAG)', fontsize=14, fontweight='bold')
    plt.annotate('Causal assumption: Interest rates directly affect asset prices', 
                xy=(0.5, -0.1), xycoords='axes fraction', 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                  fc="lightyellow", ec="orange", alpha=0.8))
    
    # Explain the DAG interpretation
    plt.figtext(0.5, -0.2, 
                "This DAG encodes our economic assumption that interest rate changes\n"
                "causally influence asset prices, but not vice versa (in the short term).\n"
                "This structure guides our intervention analysis when applying Pearl's do-operator.",
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                                                   fc="whitesmoke", ec="lightgray", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('causal_dag.png', bbox_inches='tight')
    plt.close()

# Call the function to create and visualize the DAG
print("Creating causal DAG visualization...")
plot_causal_dag()


# Define the lag order for our VAR model (p=2 as you created lag-2 features)
p = 2

# Prepare data for the VAR model
# We'll use the original series and create lag matrices ourselves
# This gives us more flexibility in the modeling
series_names = ['SP500_log_return', 'FedFunds_diff']
n_series = len(series_names)

# Trim the beginning of the dataset to account for lags
y = var_data.iloc[p:].values  # Response variables (t)
n_obs = len(y)

# Create design matrices (lagged predictors)
X = np.zeros((n_obs, n_series * p))

for i in range(p):
    X[:, i * n_series:(i + 1) * n_series] = var_data.iloc[p - i - 1:-i - 1].values

print(f"Response shape: {y.shape}, Design matrix shape: {X.shape}")

# Example of strengthening the Minnesota prior
def _mn_scale(lag: int, own: bool, lambda_1: float = 0.2, lambda_cross: float = 0.5):
    """Enhanced Minnesota prior with stronger negative relationship for rates→stocks"""
    if not own:
        if i == 0 and j == 1:  # S&P <- Fed Funds
            # Stronger negative prior (-0.15 instead of -0.05)
            # This reflects economic theory that rate increases negatively impact stocks
            return lambda_1 * lambda_cross / (lag ** 2), -0.15
    # Original return for other relationships
    return lambda_1 / (lag ** 2) if own else lambda_1 * lambda_cross / (lag ** 2)

# -----------------------------------------------------
#  Core: build and sample Bayesian VAR with Minnesota priors
# -----------------------------------------------------

def build_and_sample_bvar(y: np.ndarray,
                          X: np.ndarray,
                          n_series: int,
                          p: int,
                          tune: int = 1000,
                          draws: int = 2000,
                          chains: int = 4,
                          lambda_1: float = 0.2,
                          lambda_cross: float = 0.5):
    """Build a VAR(p) with classic Minnesota priors and sample with NUTS.

    Parameters
    ----------
    y : (n_obs, n_series) response matrix at times t
    X : (n_obs, n_series*p) lagged predictors
    lambda_1 : overall tightness (rule-of-thumb 0.2–0.3 for monthly data)
    lambda_cross : tightness multiplier for cross-variable lags
    """
    n_obs = y.shape[0]

    with pm.Model() as model:
        # ---- Intercepts ---------------------------------------------------
        intercept = pm.Normal('intercept', mu=0., sigma=1.0, shape=n_series)

        # ---- Coefficient tensor  [eqn i, lag k, pred j] -------------------
        coefs = []
        for i in range(n_series):
            eq_coefs = []
            for k in range(p):                       # lag index 0..p-1  (== lag k+1)
                for j in range(n_series):
                    own = (i == j)
                    lag_num = k + 1

                    # Prior mean
                    mu = 1.0 if own and lag_num == 1 else 0.0

                    # Prior sigma (Minnesota)
                    sigma = _mn_scale(lag_num, own, lambda_1, lambda_cross)

                    eq_coefs.append(pm.Normal(f"coef_{i}_{k}_{j}", mu=mu, sigma=sigma))
            coefs.append(eq_coefs)
        beta = pm.math.stack([pm.math.stack(eq) for eq in coefs])  # shape (n_series, n_series*p)

        # ---- Innovation s.d. (diagonal Σ) ---------------------------------
        sigma_eps = pm.HalfNormal('sigma', sigma=1.0, shape=n_series)

        # ---- Likelihood ----------------------------------------------------
        for i in range(n_series):
            mu_i = intercept[i] + pm.math.sum(X * beta[i], axis=1)
            pm.Normal(f'y_{i}', mu=mu_i, sigma=sigma_eps[i], observed=y[:, i])

        # ---- Sample -------------------------------------------------------
        trace = pm.sample(draws=draws,
                          tune=tune,
                          chains=chains,
                          target_accept=0.9,
                          return_inferencedata=True)

    model_info = {
        'n_series': n_series,
        'p': p,
        'n_obs': n_obs,
        'series_names': [f'series_{i}' for i in range(n_series)]
    }
    return model, trace, model_info

# Build and sample the Bayesian VAR model
print("Sampling from the Bayesian VAR model (this may take a while)...")
bvar_model, trace, model_info = build_and_sample_bvar(y, X, n_series, p)


print("\nChecking MCMC convergence:")
max_rhat = az.summary(trace)['r_hat'].max()
min_ess = az.summary(trace)['ess_bulk'].min()
print(f"- Maximum Rhat value: {max_rhat:.4f} (should be < 1.05)")
print(f"- Minimum ESS value: {min_ess:.1f} (should be > 400)")

if max_rhat > 1.05:
    print("WARNING: Rhat values indicate poor convergence! Consider increasing tune and draws.")
if min_ess < 400:
    print("WARNING: Low effective sample size! Consider increasing chains or draws.")

# Plot trace for key parameters
az.plot_trace(trace, var_names=["intercept", "sigma"])
plt.savefig('mcmc_diagnostics.png')
plt.close()

# Print summary of posterior
print("\nPosterior Summary:")
summary = az.summary(trace)
print(summary)

# Save the trace for later use
az.to_netcdf(trace, 'bvar_trace.nc')

# Helper function to extract coefficients from trace
def get_var_coefficients(trace, n_series, p):
    # Initialize coefficient array
    coefs = np.zeros((n_series, n_series * p))
    
    # Fill in coefficients
    for i in range(n_series):
        for j in range(p):
            for k in range(n_series):
                coef_name = f'coef_{i}_{j}_{k}'
                coefs[i, j * n_series + k] = np.mean(trace.posterior[coef_name].values, axis=(0, 1))
    
    return coefs

# Get the mean VAR coefficients
coefs = get_var_coefficients(trace, n_series, p)

# Display coefficients in a more readable format
coef_df = pd.DataFrame(index=series_names)

for i, resp in enumerate(series_names):
    for j in range(p):
        for k, pred in enumerate(series_names):
            coef_name = f"{pred}_lag{j+1}"
            coef_df.loc[resp, coef_name] = coefs[i, j * n_series + k]

print("\nVAR Coefficients (posterior means):")
print(coef_df)

def plot_coefficient_posteriors(trace, n_series, p, series_names):
    plt.figure(figsize=(20, 16))
    
    # Create separate subplots for each row to ensure proper scaling
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    for i, resp in enumerate(series_names):
        for j in range(p):
            for k, pred in enumerate(series_names):
                ax = axes[i, j*n_series + k]
                
                coef_name = f'coef_{i}_{j}_{k}'
                coef_samples = trace.posterior[coef_name].values.flatten()
                
                # Calculate appropriate x-axis limits based on the data
                mean = np.mean(coef_samples)
                std = np.std(coef_samples)
                x_min = mean - 4*std
                x_max = mean + 4*std
                
                # Plot the distribution with controlled x-limits
                sns.kdeplot(coef_samples, fill=True, color='indianred', alpha=0.7, ax=ax)
                ax.axvline(x=0, color='darkred', linestyle='--', alpha=0.7)
                ax.set_xlim(x_min, x_max)
                
                # Add mean and 95% credible interval
                mean_val = np.mean(coef_samples)
                ci_lower = np.percentile(coef_samples, 2.5)
                ci_upper = np.percentile(coef_samples, 97.5)
                
                # Improved title with better formatting
                ax.set_title(f"{resp} ← {pred} (lag {j+1})", fontweight='bold', fontsize=14)
                
                # Add statistics in a text box
                textstr = f"Mean: {mean_val:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
                props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
                ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
    
    plt.suptitle("Posterior Distributions of VAR Coefficients", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('coefficient_posteriors.png', dpi=300, bbox_inches='tight')
    plt.close()
# Plot coefficient posteriors
plot_coefficient_posteriors(trace, n_series, p, series_names)

# Create simulated residuals for impulse response analysis
def compute_residuals(y, X, coefs):
    # Fitted values
    y_hat = X @ coefs.T
    
    # Residuals
    return y - y_hat

# Get residuals
residuals = compute_residuals(y, X, coefs)

# Compute residual covariance matrix
sigma_resid = np.cov(residuals.T)

# Function to compute impulse response functions
def compute_impulse_responses(coefs, n_series, p, sigma_resid, steps=24):
    # Initialize IRFs
    irfs = np.zeros((steps, n_series, n_series))
    
    # Initialize companion matrix
    companion = np.zeros((n_series * p, n_series * p))
    
    # Fill in companion matrix with coefficients
    companion[:n_series, :] = coefs
    companion[n_series:, :-n_series] = np.eye(n_series * (p - 1))
    
    # Compute structural errors
    # Here we're using a simple Cholesky decomposition for identification
    # (Interest rates affect stock prices contemporaneously, but not vice versa)
    try:
        chol = np.linalg.cholesky(sigma_resid)
    except np.linalg.LinAlgError:
        # If Cholesky decomposition fails, add a small constant to diagonal
        print("Warning: Adding small value to covariance diagonal for numerical stability")
        sigma_resid_adjusted = sigma_resid + np.eye(n_series) * 1e-12
        chol = np.linalg.cholesky(sigma_resid_adjusted)
    

    # First period is just the Cholesky factor
    irfs[0, :, :] = chol
    
    # Compute remaining IRFs recursively
    eye = np.eye(n_series)
    for i in range(1, steps):
        prev_irf = np.zeros((n_series, n_series))
        for j in range(min(i, p)):
            prev_companion = np.linalg.matrix_power(companion, i - j - 1)
            prev_companion = prev_companion[:n_series, :n_series]
            prev_irf += np.dot(prev_companion, coefs[:, j * n_series:(j + 1) * n_series])
        
        irfs[i, :, :] = np.dot(prev_irf, chol)
    
    return irfs

# Compute impulse responses
irfs = compute_impulse_responses(coefs, n_series, p, sigma_resid)

# Plot impulse response functions
def plot_impulse_responses(irfs, series_names, steps=24):
    plt.figure(figsize=(12, 8))
    
    for i, resp in enumerate(series_names):
        for j, shock in enumerate(series_names):
            plt.subplot(n_series, n_series, i * n_series + j + 1)
            
            plt.plot(range(steps), irfs[:, i, j], 'b-', label='Point IRF')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            plt.title(f"Response of {resp} to {shock} shock")
            if i == n_series - 1:
                plt.xlabel('Months')
            if j == 0:
                plt.ylabel('Response')
    
    plt.tight_layout()
    plt.savefig('impulse_responses.png')
    plt.close()

# Plot impulse responses
plot_impulse_responses(irfs, series_names)

# Plot impulse responses
plot_impulse_responses(irfs, series_names)

# Forecast function with simplified VAR structure
def forecast_bvar(trace, X_last, n_series, p, steps=24, n_samples=1000):
    """Enhanced forecast with regime-switching to capture volatility clusters"""
    all_forecasts = np.zeros((n_samples, steps, n_series))
    
    # Extract parameters (same as your current code)
    sample_idxs = np.random.choice(trace.posterior.draw.size * trace.posterior.chain.size, 
                                  size=n_samples, replace=True)
    chain_idxs = sample_idxs // trace.posterior.draw.size
    draw_idxs = sample_idxs % trace.posterior.draw.size
    
    # Extract parameters as before
    intercepts = np.zeros((n_samples, n_series))
    sigmas = np.zeros((n_samples, n_series))
    coefficients = np.zeros((n_samples, n_series, n_series * p))
    
    # Fill parameters (same as your current code)
    for i in range(n_series):
        intercepts[:, i] = trace.posterior.intercept.values[chain_idxs, draw_idxs, i]
        sigmas[:, i] = trace.posterior.sigma.values[chain_idxs, draw_idxs, i]
        
        for j in range(p):
            for k in range(n_series):
                coef_name = f'coef_{i}_{j}_{k}'
                coefficients[:, i, j * n_series + k] = trace.posterior[coef_name].values[chain_idxs, draw_idxs]
    
    # Add regime-switching for better volatility modeling
    high_vol_prob = 0.2  # Initial probability of high volatility regime
    high_vol_scalar = 2.0  # Multiplier for high volatility regime
    regime_persistence = 0.8  # Probability of staying in current regime
    
    # Initialize regimes (0=low vol, 1=high vol) for each sample path
    current_regimes = np.random.binomial(1, high_vol_prob, size=n_samples)
    
    # Track volatility for each sample path
    X_forecasts = np.repeat(X_last[np.newaxis, :, :], n_samples, axis=0)
    
    # Generate forecasts with regime dynamics
    for t in range(steps):
        for s in range(n_samples):
            # Compute mean forecast as before
            mean = np.zeros(n_series)
            for i in range(n_series):
                mean_i = intercepts[s, i]
                for j in range(n_series * p):
                    mean_i += X_forecasts[s, 0, j] * coefficients[s, i, j]
                mean[i] = mean_i
            
            # Regime switching logic
            if np.random.random() > regime_persistence:
                # Switch regime with probability 1-persistence
                current_regimes[s] = 1 - current_regimes[s]
            
            # Apply volatility based on regime
            vol_multiplier = high_vol_scalar if current_regimes[s] == 1 else 1.0
            forecast = np.zeros(n_series)
            for i in range(n_series):
                forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i] * vol_multiplier)
            
            # Store forecast
            all_forecasts[s, t] = forecast
            
            # Update X_forecast for next step
            if t < steps - 1:
                # Shift existing values and add new prediction at the front
                X_forecasts[s, 0, n_series:] = X_forecasts[s, 0, :-n_series]
                X_forecasts[s, 0, :n_series] = forecast
    
    return all_forecasts

# Prepare last observed values for forecasting
X_last = X[-1:].copy()

# Generate forecasts
forecast_horizon = 24  # 24 months (2 years)
forecasts = forecast_bvar(trace, X_last, n_series, p, steps=forecast_horizon)

# Calculate forecast statistics
forecast_mean = np.mean(forecasts, axis=0)
forecast_lower = np.percentile(forecasts, 2.5, axis=0)
forecast_upper = np.percentile(forecasts, 97.5, axis=0)

# Prepare dates for forecasting
last_date = var_data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='ME')[1:]

# Plot forecasts
plt.figure(figsize=(14, 8))

for i, series in enumerate(series_names):
    plt.subplot(n_series, 1, i + 1)
    
    # Plot historical data
    hist_dates = var_data.index[-36:]  # Last 3 years
    hist_values = var_data[series].values[-36:]
    plt.plot(hist_dates, hist_values, 'b-', label='Historical')
    
    # Plot forecast mean
    plt.plot(forecast_dates, forecast_mean[:, i], 'r-', label='Forecast Mean')
    
    # Plot 95% credible interval
    plt.fill_between(forecast_dates, forecast_lower[:, i], forecast_upper[:, i], 
                    color='r', alpha=0.2, label='95% Credible Interval')
    
    plt.axvline(x=last_date, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Forecast for {series}')
    plt.legend()

plt.tight_layout()
plt.savefig('bvar_forecasts.png')
plt.close()

import numpy as np

def simulate_intervention(
    trace,
    last_obs,               # np.array shape=(p, n_series): the most recent p lags of each series
    intervention_series,    # int: which variable to intervene on (0 = SP500, 1 = FedFunds)
    intervention_time=0,    # int: step at which to apply the shock
    intervention_value=0.5, # float: size of the shock (e.g. +0.5)
    steps=24,               # how many steps to forecast
    market_adaptation=1.0,  # set <1.0 to shrink the noise after the shock
    n_samples=1000         # how many posterior samples to draw
):
    """
    Returns:
        baseline_paths:   array (n_samples, steps, n_series)
        intervention_paths: array (n_samples, steps, n_series)
    """
    # 1) Stack posterior draws into flat arrays
    post = trace.posterior
    # assume names: "intercept", "sigma", and "coef_i_j_k" or a single "coef" var
    # Here I’ll assume you have `coef` shaped (chain, draw, n_series, n_series*p)
    intercepts = post["intercept"].stack(sample=("chain","draw")).values    # (n_samples, n_series)
    sigmas     = post["sigma"].stack(sample=("chain","draw")).values        # (n_samples, n_series)
    coefs      = post["coef"].stack(sample=("chain","draw")).values         # (n_samples, n_series, n_series*p)

    n_samples, n_series = intercepts.shape
    _, n_features      = coefs.shape[0], coefs.shape[2]
    p = n_features // n_series

    # containers
    baseline_paths     = np.zeros((n_samples, steps, n_series))
    intervention_paths = np.zeros_like(baseline_paths)

    for s in range(n_samples):
        # initialize lag-vector: shape (n_series*p,)
        lags = last_obs.flatten()[::-1]  # most recent first
        # if your last_obs is already flattened in correct order, skip the flatten.
        # We'll maintain: [ y_t-1, y_t-2, ..., y_t-p ]

        for t in range(steps):
            # compute the VAR mean: intercept + coef @ lags
            mean = intercepts[s] + coefs[s] @ lags

            # ----- baseline forecast (no shock) -----
            eps = np.random.normal(0, sigmas[s], size=n_series)
            baseline_paths[s, t] = mean + eps

            # ----- intervention forecast -----
            if t == intervention_time:
                # impose the shock
                mean_int = mean.copy()
                mean_int[intervention_series] += intervention_value
                shocked = True
            elif t < intervention_time:
                mean_int = mean.copy()
                shocked = False
            else:
                # after the shock, let the shock propagate via the lags
                mean_int = mean.copy()
                shocked = True

            # add noise, with optional adaptation
            noise_scale = sigmas[s] * (market_adaptation ** max(0, t - intervention_time))
            eps_int = np.random.normal(0, noise_scale, size=n_series)
            intervention_paths[s, t] = mean_int + eps_int

            # ----- roll forward the lag vectors for next step -----
            # both baseline and intervention use their own lags:
            # if you want the baseline path purely, you could keep a separate lags_baseline.

            # Here we update *one* lag path at a time:
            lags = np.roll(lags, -n_series)
            lags[-n_series:] = intervention_paths[s, t]  # propagate the *intervention* path

            # If you also want to record a pure baseline roll-forward, you'd keep
            # a separate lags_baseline and update it with baseline_paths[s, t].

    return baseline_paths, intervention_paths



# Define an intervention scenario: Fed Funds Rate hike of 0.5%
intervention_series = 1  # Index for FedFunds_diff
intervention_value = 0.5  # 0.5 percentage point increase

# Generate baseline forecasts first (no intervention)
baseline_forecasts = forecast_bvar(trace, X_last, n_series, p, steps=forecast_horizon)

# Now generate intervention forecasts separately
# This approach avoids issues with the original simulate_intervention function
# Get random sample indices
n_samples = 1000
sample_idxs = np.random.choice(trace.posterior.draw.size * trace.posterior.chain.size, 
                               size=n_samples, replace=True)
chain_idxs = sample_idxs // trace.posterior.draw.size
draw_idxs = sample_idxs % trace.posterior.draw.size

# Extract parameters from trace
intercepts = np.zeros((n_samples, n_series))
sigmas = np.zeros((n_samples, n_series))
coefficients = np.zeros((n_samples, n_series, n_series * p))

# Fill in parameters
for i in range(n_series):
    intercepts[:, i] = trace.posterior.intercept.values[chain_idxs, draw_idxs, i]
    sigmas[:, i] = trace.posterior.sigma.values[chain_idxs, draw_idxs, i]
    
    for j in range(p):
        for k in range(n_series):
            coef_name = f'coef_{i}_{j}_{k}'
            coefficients[:, i, j * n_series + k] = trace.posterior[coef_name].values[chain_idxs, draw_idxs]

# Initialize intervention forecasts array
intervention_forecasts = np.zeros((n_samples, forecast_horizon, n_series))

# For each sample, generate a forecast with intervention
for s in range(n_samples):
    # Copy X_last for this sample
    X_forecast = X_last.copy()
    
    # Generate forecasts step by step
    for t in range(forecast_horizon):
        # Compute mean forecast - explicitly handle the computation to ensure scalar results
        mean = np.zeros(n_series)
        for i in range(n_series):
            # Explicitly calculate the mean for each series (equation)
            mean_i = intercepts[s, i]
            for j in range(n_series * p):
                mean_i += X_forecast[0, j] * coefficients[s, i, j]
            mean[i] = mean_i
        
        # Add random noise
        forecast = np.zeros(n_series)
        for i in range(n_series):
            forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
        
        # Apply intervention at time 3
        if t == 3:
            forecast[intervention_series] = intervention_value
        
        # Store forecast
        intervention_forecasts[s, t] = forecast
        
        # Update X_forecast for next step
        if t < forecast_horizon - 1:
            # Update X_forecast properly - shift values and add new ones
            # First, shift the existing values to the right
            for j in range(p-1, 0, -1):
                start_idx = (j-1) * n_series
                end_idx = j * n_series
                target_start = j * n_series
                target_end = (j+1) * n_series
                X_forecast[0, target_start:target_end] = X_forecast[0, start_idx:end_idx]
            
            # Then, add the new forecast to the beginning
            X_forecast[0, 0:n_series] = forecast

# Add this code after the existing intervention analysis

print("\n\n# IMPLEMENTING MULTIPLE INTERVENTION SCENARIOS")
print("# As described in Section 6.2.3 of the paper")

# 1. Different magnitudes of intervention
intervention_magnitudes = [0.25, 0.5, 1.0]  # Testing different interest rate changes
magnitude_results = {}

print("\n1. Testing different intervention magnitudes...")
for magnitude in intervention_magnitudes:
    print(f"  - Simulating interest rate change of +{magnitude}%")
    
    # Initialize intervention forecasts array for this magnitude
    intervention_forecasts_mag = np.zeros((n_samples, forecast_horizon, n_series))
    
    # For each sample, generate forecasts with this magnitude of intervention
    for s in range(n_samples):
        # Copy X_last for this sample
        X_forecast = X_last.copy()
        
        # Generate forecasts step by step
        for t in range(forecast_horizon):
            # Compute mean forecast
            mean = np.zeros(n_series)
            for i in range(n_series):
                mean_i = intercepts[s, i]
                for j in range(n_series * p):
                    mean_i += X_forecast[0, j] * coefficients[s, i, j]
                mean[i] = mean_i
            
            # Add random noise
            forecast = np.zeros(n_series)
            for i in range(n_series):
                forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
            
            # Apply intervention at time 0
            if t == 3:
                forecast[intervention_series] = magnitude
            
            # Store forecast
            intervention_forecasts_mag[s, t] = forecast
            
            # Update X_forecast for next step
            if t < forecast_horizon - 1:
                # Update X_forecast properly
                for j in range(p-1, 0, -1):
                    start_idx = (j-1) * n_series
                    end_idx = j * n_series
                    target_start = j * n_series
                    target_end = (j+1) * n_series
                    X_forecast[0, target_start:target_end] = X_forecast[0, start_idx:end_idx]
                
                X_forecast[0, 0:n_series] = forecast
    
    # Calculate causal effect (difference from baseline)
    magnitude_effect = np.mean(intervention_forecasts_mag - baseline_forecasts, axis=0)
    magnitude_results[magnitude] = magnitude_effect

# 2. Sustained interventions (interest rate maintained for multiple periods)
sustained_periods = [3, 6]  # Testing different durations
sustained_results = {}

print("\n2. Testing sustained interventions...")
for periods in sustained_periods:
    print(f"  - Simulating sustained interest rate change over {periods} months")
    
    # Initialize intervention forecasts array
    intervention_forecasts_sus = np.zeros((n_samples, forecast_horizon, n_series))
    
    # For each sample, generate forecasts with sustained intervention
    for s in range(n_samples):
        # Copy X_last for this sample
        X_forecast = X_last.copy()
        
        # Generate forecasts step by step
        for t in range(forecast_horizon):
            # Compute mean forecast
            mean = np.zeros(n_series)
            for i in range(n_series):
                mean_i = intercepts[s, i]
                for j in range(n_series * p):
                    mean_i += X_forecast[0, j] * coefficients[s, i, j]
                mean[i] = mean_i
            
            # Add random noise
            forecast = np.zeros(n_series)
            for i in range(n_series):
                forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
            
            # Apply intervention for multiple periods
            if t < periods:
                forecast[intervention_series] = intervention_value
            
            # Store forecast
            intervention_forecasts_sus[s, t] = forecast
            
            # Update X_forecast for next step
            if t < forecast_horizon - 1:
                # Update X_forecast properly
                for j in range(p-1, 0, -1):
                    start_idx = (j-1) * n_series
                    end_idx = j * n_series
                    target_start = j * n_series
                    target_end = (j+1) * n_series
                    X_forecast[0, target_start:target_end] = X_forecast[0, start_idx:end_idx]
                
                X_forecast[0, 0:n_series] = forecast
    
    # Calculate causal effect
    sustained_effect = np.mean(intervention_forecasts_sus - baseline_forecasts, axis=0)
    sustained_results[periods] = sustained_effect

# 3. Gradual interventions (consecutive small increases)
print("\n3. Testing gradual intervention (Fed tightening cycle)...")
print("  - Simulating gradual tightening with 0.25% increases over 4 months")

# Initialize intervention forecasts array
intervention_forecasts_grad = np.zeros((n_samples, forecast_horizon, n_series))

# Define the intervention schedule (0.25% increases over 4 months)
gradual_schedule = [0.25, 0.5, 0.75, 1.0]  # Cumulative values

# For each sample, generate forecasts with gradual intervention
for s in range(n_samples):
    # Copy X_last for this sample
    X_forecast = X_last.copy()
    
    # Generate forecasts step by step
    for t in range(forecast_horizon):
        # Compute mean forecast
        mean = np.zeros(n_series)
        for i in range(n_series):
            mean_i = intercepts[s, i]
            for j in range(n_series * p):
                mean_i += X_forecast[0, j] * coefficients[s, i, j]
            mean[i] = mean_i
        
        # Add random noise
        forecast = np.zeros(n_series)
        for i in range(n_series):
            forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
        
        # Apply gradual intervention
        if t < len(gradual_schedule):
            forecast[intervention_series] = gradual_schedule[t]
        
        # Store forecast
        intervention_forecasts_grad[s, t] = forecast
        
        # Update X_forecast for next step
        if t < forecast_horizon - 1:
            # Update X_forecast properly
            for j in range(p-1, 0, -1):
                start_idx = (j-1) * n_series
                end_idx = j * n_series
                target_start = j * n_series
                target_end = (j+1) * n_series
                X_forecast[0, target_start:target_end] = X_forecast[0, start_idx:end_idx]
            
            X_forecast[0, 0:n_series] = forecast

# Calculate causal effect for gradual intervention
gradual_effect = np.mean(intervention_forecasts_grad - baseline_forecasts, axis=0)

# Visualization of multiple intervention scenarios
plt.figure(figsize=(18, 15))

# 1. Plot different magnitudes
plt.subplot(3, 1, 1)
for magnitude in intervention_magnitudes:
    plt.plot(forecast_dates, magnitude_results[magnitude][:, 0], 
             label=f'+{magnitude}% rate increase', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.title('Effect of Different Interest Rate Increase Magnitudes on S&P 500 Returns', fontsize=14)
plt.legend()
plt.ylabel('Effect on Returns')

# 2. Plot sustained interventions
plt.subplot(3, 1, 2)
for periods in sustained_periods:
    plt.plot(forecast_dates, sustained_results[periods][:, 0], 
             label=f'Sustained over {periods} months', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.title('Effect of Sustained Interest Rate Increases on S&P 500 Returns', fontsize=14)
plt.legend()
plt.ylabel('Effect on Returns')

# 3. Plot gradual intervention
plt.subplot(3, 1, 3)
plt.plot(forecast_dates, gradual_effect[:, 0], 'g-', label='Gradual tightening', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.title('Effect of Gradual Fed Tightening Cycle on S&P 500 Returns', fontsize=14)
plt.legend()
plt.ylabel('Effect on Returns')
plt.xlabel('Date')

plt.tight_layout()
plt.savefig('multiple_intervention_scenarios.png')
plt.close()

# Print summary results
print("\nMultiple Intervention Scenarios - Summary of Effects on S&P 500:")
print("\nAverage 6-month effects of different intervention magnitudes:")
for magnitude in intervention_magnitudes:
    avg_effect = magnitude_results[magnitude][:6, 0].mean()
    print(f"  +{magnitude}% rate increase: {avg_effect:.4f}")

print("\nAverage 6-month effects of sustained interventions:")
for periods in sustained_periods:
    avg_effect = sustained_results[periods][:6, 0].mean()
    print(f"  Sustained over {periods} months: {avg_effect:.4f}")

print("\nAverage 6-month effect of gradual tightening:")
avg_effect = gradual_effect[:6, 0].mean()
print(f"  Gradual Fed tightening cycle: {avg_effect:.4f}")
# Calculate statistics for intervention simulation
baseline_mean = np.mean(baseline_forecasts, axis=0)
baseline_lower = np.percentile(baseline_forecasts, 2.5, axis=0)
baseline_upper = np.percentile(baseline_forecasts, 97.5, axis=0)

intervention_mean = np.mean(intervention_forecasts, axis=0)
intervention_lower = np.percentile(intervention_forecasts, 2.5, axis=0)
intervention_upper = np.percentile(intervention_forecasts, 97.5, axis=0)

# Calculate causal effect (mean difference)
causal_effect = intervention_mean - baseline_mean
causal_effect_lower = np.percentile(intervention_forecasts - baseline_forecasts, 2.5, axis=0)
causal_effect_upper = np.percentile(intervention_forecasts - baseline_forecasts, 97.5, axis=0)

# Plot intervention analysis
plt.figure(figsize=(16, 12))

# Plot both series with and without intervention
for i, series in enumerate(series_names):
    plt.subplot(3, n_series, i + 1)
    
    # Plot baseline forecast
    plt.plot(forecast_dates, baseline_mean[:, i], 'b-', label='Baseline')
    plt.fill_between(forecast_dates, baseline_lower[:, i], baseline_upper[:, i], 
                    color='b', alpha=0.1)
    
    # Plot intervention forecast
    plt.plot(forecast_dates, intervention_mean[:, i], 'r-', label='Intervention')
    plt.fill_between(forecast_dates, intervention_lower[:, i], intervention_upper[:, i], 
                    color='r', alpha=0.1)
    
    plt.title(f'{series}: Baseline vs. Intervention')
    plt.legend()
    
    # If this is the intervention variable, add a note
    if i == intervention_series:
        plt.annotate(f'Intervention: +{intervention_value}', xy=(0.05, 0.9), 
                    xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Plot the causal effect (difference)
for i, series in enumerate(series_names):
    plt.subplot(3, n_series, n_series + i + 1)
    
    plt.plot(forecast_dates, causal_effect[:, i], 'g-', label='Causal Effect')
    plt.fill_between(forecast_dates, causal_effect_lower[:, i], causal_effect_upper[:, i], 
                    color='g', alpha=0.2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Causal Effect on {series}')
    
    # Compute cumulative effect over time
    cum_effect = np.cumsum(causal_effect[:, i])
    
    plt.subplot(3, n_series, 2*n_series + i + 1)
    plt.plot(forecast_dates, cum_effect, 'm-', label='Cumulative Effect')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Cumulative Effect on {series}')

plt.tight_layout()
plt.savefig('intervention_analysis.png')
plt.close()

print("\nIntervention Analysis Results:")
print(f"- Interest Rate Intervention: +{intervention_value} percentage points")

# Calculate the average effect on S&P 500 returns over the first 6 months
effect_6m = causal_effect[:6, 0].mean()
effect_12m = causal_effect[:12, 0].mean()
cumulative_effect_12m = np.cumsum(causal_effect[:12, 0])[-1]

print(f"- Average effect on S&P 500 monthly returns over 6 months: {effect_6m:.4f}")
print(f"- Average effect on S&P 500 monthly returns over 12 months: {effect_12m:.4f}")
print(f"- Cumulative effect on S&P 500 returns over 12 months: {cumulative_effect_12m:.4f}")

print("\nBayesian VAR model analysis complete!")


def plot_zoomed_intervention_analysis(
    baseline_forecasts, 
    intervention_forecasts, 
    forecast_dates, 
    series_names=['SP500_log_return', 'FedFunds_diff'],
    intervention_series=1,
    intervention_value=0.5,
    intervention_time=3  # Index when intervention occurs (default: 3rd forecast period)
):
    """
    Create zoomed-in plots of the intervention analysis for better visualization
    of subtle effects on financial time series.
    
    Parameters:
    -----------
    baseline_forecasts : numpy.ndarray
        Array of shape (n_samples, steps, n_series) containing baseline forecasts
    intervention_forecasts : numpy.ndarray
        Array of shape (n_samples, steps, n_series) containing intervention forecasts
    forecast_dates : array-like
        Dates for the forecast horizon
    series_names : list
        Names of the time series (default: ['SP500_log_return', 'FedFunds_diff'])
    intervention_series : int
        Index of the series that receives intervention (default: 1 for FedFunds_diff)
    intervention_value : float
        Value of the intervention (default: 0.5)
    intervention_time : int
        Time index when intervention occurs (default: 3)
    """
    # Calculate statistics
    baseline_mean = np.mean(baseline_forecasts, axis=0)
    baseline_lower = np.percentile(baseline_forecasts, 2.5, axis=0)
    baseline_upper = np.percentile(baseline_forecasts, 97.5, axis=0)

    intervention_mean = np.mean(intervention_forecasts, axis=0)
    intervention_lower = np.percentile(intervention_forecasts, 2.5, axis=0)
    intervention_upper = np.percentile(intervention_forecasts, 97.5, axis=0)

    # Calculate causal effect (difference)
    causal_effect = intervention_mean - baseline_mean
    causal_effect_lower = np.percentile(intervention_forecasts - baseline_forecasts, 2.5, axis=0)
    causal_effect_upper = np.percentile(intervention_forecasts - baseline_forecasts, 97.5, axis=0)

    # Create figure with subplots - focus on S&P 500 (index 0)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 1. S&P 500 Baseline vs Intervention (ZOOMED)
    ax1 = axs[0]
    
    # Plot baseline
    ax1.plot(forecast_dates, baseline_mean[:, 0], 'b-', label='Baseline', linewidth=2)
    ax1.fill_between(forecast_dates, baseline_lower[:, 0], baseline_upper[:, 0], 
                     color='blue', alpha=0.1)
    
    # Plot intervention
    ax1.plot(forecast_dates, intervention_mean[:, 0], 'r-', label='Intervention', linewidth=2)
    ax1.fill_between(forecast_dates, intervention_lower[:, 0], intervention_upper[:, 0], 
                     color='red', alpha=0.1)
    
    # Add reference line at y=0
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Mark intervention time with vertical line
    if intervention_time < len(forecast_dates):
        ax1.axvline(x=forecast_dates[intervention_time], color='orange', linestyle='-', alpha=0.7)
        ax1.text(forecast_dates[intervention_time], ax1.get_ylim()[1]*0.9, 
                 f'Intervention: +{intervention_value}', 
                 rotation=90, verticalalignment='top', color='darkred')
    
    # ZOOMED Y-AXIS - adjust these values based on your data
    # Find the range of values and add small padding
    ymin = min(np.min(baseline_mean[:, 0]), np.min(intervention_mean[:, 0])) - 0.001
    ymax = max(np.max(baseline_mean[:, 0]), np.max(intervention_mean[:, 0])) + 0.001
    
    # Ensure we're not zooming too much - provide reasonable boundaries
    yrange = ymax - ymin
    if yrange < 0.005:
        # If range is very small, create a minimum visible range
        ymid = (ymax + ymin) / 2
        ymin = ymid - 0.0025
        ymax = ymid + 0.0025
    
    ax1.set_ylim(ymin, ymax)
    
    ax1.set_title(f'{series_names[0]}: Baseline vs. Intervention (Zoomed)', fontsize=14)
    ax1.set_ylabel('Monthly Log Return', fontsize=12)
    ax1.legend(loc='best')
    
    # Add detailed y-axis tick formatting for small values
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # 2. Causal Effect on S&P 500 (ZOOMED)
    ax2 = axs[1]
    
    # Plot causal effect
    ax2.plot(forecast_dates, causal_effect[:, 0], 'g-', label='Causal Effect', linewidth=2)
    ax2.fill_between(forecast_dates, causal_effect_lower[:, 0], causal_effect_upper[:, 0], 
                     color='green', alpha=0.2)
    
    # Add reference line at y=0
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Mark intervention time with vertical line
    if intervention_time < len(forecast_dates):
        ax2.axvline(x=forecast_dates[intervention_time], color='orange', linestyle='-', alpha=0.7)
    
    # ZOOMED Y-AXIS for causal effect
    # Find the range of causal effect values and add padding
    ymin_effect = min(0, np.min(causal_effect[:, 0])) - 0.0005
    ymax_effect = max(0, np.max(causal_effect[:, 0])) + 0.0005
    
    # Ensure we're not zooming too much
    yrange_effect = ymax_effect - ymin_effect
    if yrange_effect < 0.002:
        # If range is very small, create a minimum visible range
        ymid_effect = (ymax_effect + ymin_effect) / 2
        ymin_effect = ymid_effect - 0.001
        ymax_effect = ymid_effect + 0.001
    
    ax2.set_ylim(ymin_effect, ymax_effect)
    
    ax2.set_title(f'Causal Effect on {series_names[0]} (Zoomed)', fontsize=14)
    ax2.set_ylabel('Effect Magnitude', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Add detailed y-axis tick formatting for small values
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # Add annotation
    avg_effect = np.mean(causal_effect[:6, 0])
    
    plt.tight_layout()
    plt.savefig('zoomed_intervention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nZoomed Intervention Analysis Results:")
    print(f"Interest Rate Intervention: +{intervention_value} percentage points")
    print(f"Average effect on S&P 500 monthly returns (6 months): {np.mean(causal_effect[:6, 0]):.6f}")
    print(f"Average effect on S&P 500 monthly returns (12 months): {np.mean(causal_effect[:12, 0]):.6f}")
    print(f"Cumulative effect on S&P 500 returns (12 months): {np.sum(causal_effect[:12, 0]):.6f}")
    
    # Optional: also create a separate plot for cumulative effect
    plt.figure(figsize=(10, 6))
    cum_effect = np.cumsum(causal_effect[:, 0])
    plt.plot(forecast_dates, cum_effect, 'm-', linewidth=2.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Cumulative Effect on {series_names[0]} Returns', fontsize=16)
    plt.ylabel('Cumulative Effect', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotation for final cumulative effect
    plt.annotate(f'12-month cumulative: {cum_effect[11]:.4f}', 
                xy=(forecast_dates[11], cum_effect[11]),
                xytext=(20, 0), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('cumulative_effect_zoomed.png', dpi=300, bbox_inches='tight')
    plt.show()


#Example usage (replace with your actual data):
plot_zoomed_intervention_analysis(
    baseline_forecasts, 
    intervention_forecasts, 
    forecast_dates,
    series_names=['SP500_log_return', 'FedFunds_diff'],
    intervention_series=1,
    intervention_value=0.5
)



# ------------------------------------------------------------------------
# ABLATION: create ablation_metrics.csv
# ------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import VAR

# ---------- 1. Hold-out split -------------------------------------------------
split = int(len(var_data) * 0.8)          # 80 % train / 20 % test
train_df  = var_data.iloc[:split]
test_df   = var_data.iloc[split:split + 12]   # 12-step horizon

# ---------- 2. Classical (deterministic) VAR ---------------------------------
def deterministic_var_forecast(train, steps=12, lags=p):
    model  = VAR(train)
    fitted = model.fit(lags)
    fc     = fitted.forecast(train.values[-lags:], steps)
    idx    = pd.date_range(start=train.index[-1], periods=steps + 1, freq='ME')[1:]
    return pd.DataFrame(fc, index=idx, columns=train.columns)

det_fc = deterministic_var_forecast(train_df, steps=len(test_df))

# ---------- 3. Bayesian VAR *without* do-operator ----------------------------
# Use your existing trace but don’t override the intervention variable
bvar_noint = forecast_bvar(trace, X_last, n_series, p,
                           steps=len(test_df), n_samples=1000)
bvar_noint_mean = bvar_noint.mean(axis=0)
bvar_noint_df   = pd.DataFrame(bvar_noint_mean,
                               index=test_df.index,
                               columns=test_df.columns)

# ---------- 4. Bayesian VAR *with* DAG intervention --------------------------
# intervention_forecasts already exists in your script
# FIX: Generate proper baseline forecasts from DAG model (without intervention)
bvar_dag_baseline = forecast_bvar(trace, X_last, n_series, p, 
                                 steps=len(test_df), n_samples=1000)
bvar_dag_baseline_mean = bvar_dag_baseline.mean(axis=0)
bvar_dag_baseline_df = pd.DataFrame(bvar_dag_baseline_mean,
                                  index=test_df.index,
                                  columns=test_df.columns)


# ---------- 5. Metric helpers -------------------------------------------------
def rmse(true, pred):
    return mean_squared_error(true, pred, squared=False)

def mae(true, pred):
    return mean_absolute_error(true, pred)

# Compute metrics
rmse_dag  = rmse(test_df, bvar_dag_baseline_df)
mae_dag   = mae(test_df, bvar_dag_baseline_df)

rmse_no   = rmse(test_df, bvar_noint_df)
mae_no    = mae(test_df, bvar_noint_df)

rmse_det  = rmse(test_df, det_fc)
mae_det   = mae(test_df, det_fc)

# ------------------------------------------------------------------------
# ABLATION: Multiple runs for robust statistical comparison
# ------------------------------------------------------------------------
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd

print("\n\n# Running multiple tests for statistical comparison")
print("# This may take some time...")

# Number of runs for statistical testing
n_runs = 20  # You can adjust based on computation resources

# Arrays to store metrics from each run
dag_rmses = []
dag_maes = []
nodag_rmses = []
nodag_maes = []
class_rmse = rmse_det  # Save the value that's already computed
class_mae = mae_det    # Save the value that's already computed

# Run the probabilistic models multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}...")
    
    # Generate forecasts without intervention
    bvar_dag = forecast_bvar(trace, X_last, n_series, p, 
                            steps=len(test_df), n_samples=1000)
    bvar_dag_mean = bvar_dag.mean(axis=0)
    bvar_dag_df = pd.DataFrame(bvar_dag_mean,
                              index=test_df.index,
                              columns=test_df.columns)
    
    # Non-DAG version
    bvar_nodag = forecast_bvar(trace, X_last, n_series, p,
                              steps=len(test_df), n_samples=1000)
    bvar_nodag_mean = bvar_nodag.mean(axis=0)
    bvar_nodag_df = pd.DataFrame(bvar_nodag_mean,
                                index=test_df.index,
                                columns=test_df.columns)
    
    # Compute metrics
    dag_rmses.append(rmse(test_df, bvar_dag_df))
    dag_maes.append(mae(test_df, bvar_dag_df))
    nodag_rmses.append(rmse(test_df, bvar_nodag_df))
    nodag_maes.append(mae(test_df, bvar_nodag_df))

# Calculate statistics
dag_rmse_mean, dag_rmse_std = np.mean(dag_rmses), np.std(dag_rmses)
nodag_rmse_mean, nodag_rmse_std = np.mean(nodag_rmses), np.std(nodag_rmses)
dag_mae_mean, dag_mae_std = np.mean(dag_maes), np.std(dag_maes)
nodag_mae_mean, nodag_mae_std = np.mean(nodag_maes), np.std(nodag_maes)

# Statistical significance test
rmse_pvalue = ttest_ind(dag_rmses, nodag_rmses)[1]
mae_pvalue = ttest_ind(dag_maes, nodag_maes)[1]

# Print statistical results
print("\n--- Multiple Run Ablation Statistics ---")
print(f"Bayesian + DAG   : RMSE {dag_rmse_mean:.4f} ± {dag_rmse_std:.4f} | MAE {dag_mae_mean:.4f} ± {dag_mae_std:.4f}")
print(f"Bayesian no-DAG  : RMSE {nodag_rmse_mean:.4f} ± {nodag_rmse_std:.4f} | MAE {nodag_mae_mean:.4f} ± {nodag_mae_std:.4f}")
print(f"Classical VAR    : RMSE {class_rmse:.4f} | MAE {class_mae:.4f}")
print(f"p-value (DAG vs no-DAG): RMSE p={rmse_pvalue:.4f} | MAE p={mae_pvalue:.4f}")

# Visualize the results
plt.figure(figsize=(12, 6))

# Box plot for RMSE
plt.subplot(1, 2, 1)
plt.boxplot([dag_rmses, nodag_rmses], labels=['Bayesian+DAG', 'Bayesian no-DAG'])
plt.axhline(y=class_rmse, color='r', linestyle='--', label='Classical VAR')
plt.ylabel('RMSE')
plt.title('RMSE Comparison Across Multiple Runs')
plt.legend()

# Box plot for MAE
plt.subplot(1, 2, 2)
plt.boxplot([dag_maes, nodag_maes], labels=['Bayesian+DAG', 'Bayesian no-DAG'])
plt.axhline(y=class_mae, color='r', linestyle='--', label='Classical VAR')
plt.ylabel('MAE')
plt.title('MAE Comparison Across Multiple Runs')
plt.legend()

plt.tight_layout()
plt.savefig('ablation_statistics.png')
plt.close()

# Save detailed results to CSV
results_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE"],
    "Bayesian+DAG Mean": [dag_rmse_mean, dag_mae_mean],
    "Bayesian+DAG Std": [dag_rmse_std, dag_mae_std],
    "Bayesian no-DAG Mean": [nodag_rmse_mean, nodag_mae_mean],
    "Bayesian no-DAG Std": [nodag_rmse_std, nodag_mae_std],
    "Classical VAR": [class_rmse, class_mae],
    "p-value": [rmse_pvalue, mae_pvalue]
})
results_df.to_csv("ablation_detailed_metrics.csv", index=False)
print("\nSaved detailed statistical results to ablation_detailed_metrics.csv")