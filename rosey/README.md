<!-- ruff: noqa -->
<!-- linting: ignore -->
# send_time_p13n

# GOAL:

Predict the ideal send time for each contact to maximize probability of interaction

# HYPOTHESIS:

The closer an email is to the top of a contact's inbox the more probable it is that interact.

# LEGACY MODEL:

Creates a pdf of historical interactions for each contact. Essentially a count is performed of how often a contact has interacted (opens or clicks) with emails in the past, these counts populate a pdf that represents every possible hour of the week (168 hours). This count is normalized to create probabilities that a custom `argmax` function ingests and selects send times from.
    
    NOTE: If a contact does not meet a certain threshold of interactions in the lookback window, a default `mode` distribution is used for that contact.

# NEW MODEL:

A catboost classifier is trained on enriched historical interaction data for each contact. The features used from enrichment are provided by the customer or publicly available (ie H3 features).

# WHY SHOULD THE NEW MODEL BE BETTER?

- Generalized learning from `high` interaction users to `low` to `no` interaction users
- Minimize lookback window due to learning from population level data
- Go beyond interaction data to make predictions (ie demographic data)

# RESULTS OF RECENT AB TEST

We lost :-( (I told everyone that it was because of Stephen....)

# Things I need help on:

- Auto feature selection to replace manual EDA
- How should I evaluate a model offline? Currently use LogLoss and ROC curves.
- Model selection, is Catboost the correct model? I've been told the idea is train one and then try to beat it.

# AUTO FEATURE SELECTION PROCESS

AIM FOR CAUSALLY UPSTREAM OF THE VARIABLE

- Ingest contact features provided by customer
- Filter out based on `feasible_dtypes`
- Filter out based on `null_frac_threshold` (0.8 threshold)
    - MAKE SURE THE FEATURES DROPPED ARE OBVIOUSLY BAD, WHAT ABOUT FEATURES CLOSE TO THE THRESHOLD?
    - Rules of thumb can be misleading
- Filter out highly correlated features (0.95 threshold)
    - Tree based models are not unstable with highly correlated features, but feature importance can be messed up
- Use `catboost.select_features()` functionality, basically trains a model and filters features using `feature_importance`
- Select the `top_n` features

# OFFLINE TESTING:

- What is the AUC on the low interacation audience?
- Actually check when the email was sent vs when we thought it should. (Plot of residual of when it was actually sent vs desired send time.)
    CONFIRM IT IS ACTUALLY A MODEL PROBLEM! (Compliance Testing)
- Customer Splitting (hold out customers that the model never sees during training)
- Hold out the last week of historical data to test on. (Need to match up all the datetime features properly)

# TODO:

- Catboost Baseline functionality (see catboost website)
- Predict click_hour?
- Data Reduction?
- Include Mode Distribution as an input feature to the model
- Add conversion probability to the `send_to_interaction` distribution plots
- Check on `data_drift` between training, test and live AB
- Check on contacts that got emails from both the legacy and the new model.

# REPORT BACK 2ish WEEKs
