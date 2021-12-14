"""
Emma Hegermiller
Task:
Build a predictive model leveraging the historical data, additional data insights is a plus.
Model should forecast vin price based on the vehicle attributes provided above (i.e. make, model, make year, mileage, trim..)

Dataset information
index_pandas,
vin,
days_on_market,
listing_date_begin,
listing_date_end,
year,
make,
model,
trim,
body_style,
transmission,
mileage,
accident_count,
fuel_economy_city,
fuel_economy_highway,
exterior_color,
msrp,
is_cpo,
seller_city,
seller_state,
seller_type,
seller_is_franchise_dealer,
seller_is_online_only,
seller_ships_nationwide,
list_price, 
shift_buyer_region - not a velichle attribute

Steps:

Exploratory data analysis
    Count per categorical
    Price distribution overall
    Price distribution per categorical
Preprocessing
    drop rows with vin and list_price na
    drop columns that will not be used in prediction
    change types of categories
Calculate dataset values for imputation
Split into train/test 80/20
    based on unique vins
    check vin overlap
Feature engineering
    Group by observations
    Imputation
        median for numeric
		True/False for boolean depending on which is populated
        mode for categorical low subcategory count
        'Other' for categorical high subcategory count
    Handle dates
    Split variable
    Handle Outliers
    Normalization
    One hot encoding
    Extra features - pending
    Text feature extraction - pending
Model: Ridge Regression

Model performance: Root Mean Squared Logarithmic Error(RMSLE)

Qutstanding Questions:
What does each row in the dataset signify? 
What is the unique observation for prediction?
Given the instructions, should the predictions be made for unique vin numbers? 
Would you consider seller variables vehicle attributes?  
Also, what is the variable definition for "is_cpo"?
Prediction timeline: prediction at listing or days after listing?

"""