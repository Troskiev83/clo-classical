from scipy.optimize import linprog
import numpy as np

def run(input_data, solver_params, extra_arguments):
    # Extract data from input
    loans = input_data["loans"]
    industry_concentration_limit = input_data["industry_concentration_limit"]
    country_concentration_limit = input_data["country_concentration_limit"]
    credit_risk_threshold = input_data["credit_risk_threshold"]
    excluded_sectors = input_data["excluded_sectors"]
    total_exposure_limit = input_data["total_exposure_limit"]
    
    # Number of loans
    n_loans = len(loans)
    
    # Objective function: maximize expected returns
    c = -np.array([loan['expected_return'] for loan in loans])  # We negate to maximize using linprog
    
    # Constraints initialization
    A_ub = []
    b_ub = []
    
    # Total exposure constraint (optional, if provided)
    exposure = np.array([loan['exposure'] for loan in loans])
    if total_exposure_limit:
        A_ub.append(exposure)
        b_ub.append(total_exposure_limit)
    
    # Industry concentration limits
    for j in range(len(loans[0]['industries'])):
        industry_exposure = np.array([loan['exposure'] * loan['industries'][j] for loan in loans])
        A_ub.append(industry_exposure)
        b_ub.append(industry_concentration_limit * sum(exposure))
    
    # Country concentration limits
    for k in range(len(loans[0]['countries'])):
        country_exposure = np.array([loan['exposure'] * loan['countries'][k] for loan in loans])
        A_ub.append(country_exposure)
        b_ub.append(country_concentration_limit * sum(exposure))
    
    # Exclude certain sectors
    for l in excluded_sectors:
        sector_exclusion = np.array([loan['sectors'][l] for loan in loans])
        A_ub.append(sector_exclusion)
        b_ub.append(0)  # Ensure excluded sectors are not included

    # Credit risk threshold constraint
    credit_risk_constraint = np.array([loan['credit_risk_rating'] < credit_risk_threshold for loan in loans])
    A_ub.append(-credit_risk_constraint)  # Ensure loans below threshold are selected
    b_ub.append(0)
    
    # Binary bounds for decision variables (x_i in {0, 1})
    bounds = [(0, 1) for _ in range(n_loans)]
    
    # Solve the problem using linear programming
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Prepare result
    selected_loans = [int(x) for x in result.x]
    
    # Construct the result dictionary
    res = {
        "selected_loans": selected_loans,
        "total_expected_return": -result.fun,  # Remember to negate the result back
        "total_exposure": sum(exposure[i] for i in range(n_loans) if selected_loans[i]),
        "success": result.success
    }
    
    return res