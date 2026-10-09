# House Price Prediction: Hyderabad, India

Regression of apartment prices in Hyderabad from size, bedrooms, amenities and location. Locality names are turned
into latitude/longitude with geocoding, the features go through an imputation, scaling and PCA pipeline, and KNN,
Random Forest and Gradient Boosting are combined in a voting ensemble.

Coursework for the ITI AI-Pro diploma (2021–22): assignment 1 of the machine learning course.

## Data

`Hyderabad.csv` from [Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/ruchi798/housing-prices-in-metropolitan-areas-of-india)
(Kaggle): 2,518 listings, 40 columns (price in INR, area, locality, bedrooms, resale flag and 35 binary amenity
flags). In the source data `9` means "unknown" for the amenity flags; those 84 rows are dropped, leaving 2,434.

## Approach

1. 80/20 split stratified on binned `Area`: 1,947 training rows, 487 test rows.
2. Locality names are geocoded to latitude/longitude with geopy's Nominatim client. This is slow, so it was run once
   and the results are cached in `latitude_*.csv` / `longitude_*.csv`; the notebook reads those files.
   Locations that could not be geocoded (130 training, 32 test) are set to missing and filled with the median.
3. Pipeline: median imputation, standard scaling, PCA to 6 components.
4. Models tuned with `GridSearchCV` on RMSLE: KNN, Random Forest and Gradient Boosting, then a `VotingRegressor`
   that averages the three.

## Results

Root mean squared log error (lower is better):

| Model | 6-fold CV RMSLE (train) |
|---|---|
| KNN (k=4) | 0.307 ± 0.012 |
| Gradient Boosting (200 trees, depth 7, lr 0.21) | 0.287 ± 0.010 |
| Random Forest (230 trees, depth 26) | 0.281 ± 0.016 |
| Voting (KNN + RF + GBM) | **0.275 ± 0.012** |

The voting ensemble scores **RMSLE 0.227 on the 487 held-out test rows**.

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy missingno scikit-learn xgboost geopy joblib
jupyter notebook HousesPrice_Prediction_india.ipynb
```

All CSVs are in the repo, so the notebook runs without calling the geocoding service.

## Notes

- Nominatim was queried with the locality name only, without "Hyderabad", so some names resolved to the wrong place:
  463 of the 1,947 cached training coordinates fall outside a ±0.5° box around Hyderabad, a few of them in other
  countries. Adding the city and state to the query would fix most of these.
- PCA to 6 components is applied before every model, tree models included; the notebook does not compare against
  the raw features.
- There is no simple baseline (for example, price per square foot by locality) to put the RMSLE in context.
