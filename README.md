# Shapley-based Causal Explanations of AI Models for Electricity Price Forecasting

## Data Availability
The data that was used is publicly available via the following links:
- ENTSO-E https://transparency.entsoe.eu/
- carbon prices https://carbonpricingdashboard.worldbank.org/compliance/price
- gas prices https://fred.stlouisfed.org/series/PNGASEUUSDM
- oil prices https://fred.stlouisfed.org/series/DCOILBRENTEU

## Shapley Flow repository
The Shapley flow implementation is available at https://github.com/nathanwang000/Shapley-Flow

## Abstract
The increasing complexity of modern machine learning models presents
a challenge to their interpretability. In response, the field of Explainable
Artificial Intelligence (XAI) has emerged as a means of making these
so-called black-box models explainable and providing insights into the
relevance of input features. Previous research in the energy system domain
has extensively utilized Shapley additive explanations (SHAP), although
its inherent feature independence assumption can lead to difficulties
when explaining models that rely on data that violates this assumption.
Consequently, several approaches based on Shapley values have been
proposed with the aim of incorporating causal knowledge and improving
SHAP. As accurate electricity price forecasting is crucial for maintaining
the stability of the electricity grid, we develop machine learning models
based on neural networks and gradient boosted trees for predicting day-
ahead electricity prices in France based on a set of publicly available
techno-economic features. We generate explanations using the novel
Shapley flow approach, through which we uncover hidden relationships
that are not visible with the commonly employed SHAP framework.
