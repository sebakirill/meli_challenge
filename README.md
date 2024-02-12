# Beauty Vertical First Purchase Probability Prediction Project


The goal of this project is to develop a predictive model that can estimate the probability of a customer making their first purchase in the Beauty vertical. Machine learning and data processing techniques will be used to create an accurate and robust model that can help identify conversion opportunities and improve marketing and sales strategies.

## Development

In the development of this project, the following methods and techniques were employed:

- **Feature Selection with XGBoost:** The XGBoost algorithm was applied to select the most relevant features that influence the probability of purchase in the Beauty vertical. This helped to reduce the dimensionality of the data and improve model efficiency.
- **Encoding:** Encoding techniques were implemented to convert categorical variables into numerical representations that could be used by machine learning models. Encodings such as One-Hot Encoding and Frequency Encoding were employed to effectively handle categorical variables.
- **Imputing:** Mean imputation and categorical imputation techniques were applied to handle missing values in the dataset, ensuring that the models could be trained on complete data.

## Models Used
Several machine learning models were evaluated and compared to predict the probability of purchase in the Beauty vertical. The models used include:
- LightGBM
- Logistic Regression
- Random Forest
- XGBoost

## Folder Structure

```bash
meli_challenge/
├── conf/
│   ├── data/
│   ├── feature_selectio/
│   ├── models/
│   ├── preprocess/
│   ├── save_selected_columns/
│
├── data/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── preprocess/
│   ├── utils/
│
├── utils/
│
├── README.md
```

## Execution

## Execution
1. Clone this repository to your local machine.
2. Install the necessary dependencies using 
```bash 
pip install -r requirements.txt 
```
3. Adjust the model configuration in the /conf directory as needed.
4. Multirun with hydra 
```bash 
python main.py  -m +models=lgbm_clas,xgb_clas,rad_for,log_reg +preprocess.encoding=freq_encoder,one_hot_encoder 
```



