# Model Card

For additional information, see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
- **Model type**: Supervised Learning Classifier (Random Forest)
- **Developer**: Western Governor's University/Eric Bailey
- **Framework**: scikit-learn
- **Version**: 1.0
- **Contact**: ebai147@wgu.edu

## Intended Use
- **Primary use case**: Predicts whether an individual's income exceeds $50K based on demographic and employment features.
- **Users**: Data scientists, analysts, researchers.
- **Downstream applications**: Workforce analytics, socio-economic studies.
- **Out-of-scope applications**: High-stakes decision-making such as lending or hiring without further validation and fairness analysis.

## Training Data
- **Source**: UCI Adult Dataset
- **Size**: 48,842 samples with 14 features including age, workclass, education, marital status, and occupation.
- **Bias Considerations**: The dataset reflects US Census data from the early 1990s and may not generalize well to other time periods, geographies, or demographics.

## Evaluation Data
- **Source**: 20% test split from the UCI Adult Dataset.
- **Size**: ~9,700 samples.
- **Differences from training**: Drawn from the same population and distribution as the training data.

## Metrics
- **Overall Metrics**:  
  - Precision: 0.7376  
  - Recall: 0.6404  
  - F1: 0.6856  

- **Slice-Based Metrics**:  
  The model's performance varies significantly across categorical slices. Below are examples of the performance metrics for key categories:

  ### Workclass
  - `Federal-gov`: Precision = 0.7971, Recall = 0.7857, F1 = 0.7914  
  - `Private`: Precision = 0.7376, Recall = 0.6404, F1 = 0.6856  
  - `Without-pay`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000  

  ### Education
  - `Bachelors`: Precision = 0.7523, Recall = 0.7289, F1 = 0.7404  
  - `Masters`: Precision = 0.8271, Recall = 0.8551, F1 = 0.8409  
  - `Preschool`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000  

  ### Sex
  - `Male`: Precision = 0.7445, Recall = 0.6599, F1 = 0.6997  
  - `Female`: Precision = 0.7229, Recall = 0.5150, F1 = 0.6015  

  For a complete breakdown of all slices, see `slice_output.txt`.

## Ethical Considerations
- **Fairness**: Performance disparities exist across demographic slices such as sex and workclass, which may lead to biased outcomes.
- **Privacy**: The model uses sensitive personal attributes like sex and race. Users must ensure data privacy regulations (e.g., GDPR) are adhered to.
- **Risks**: Misuse of the model for decisions affecting individuals' livelihoods could exacerbate inequalities.

## Caveats and Recommendations
- The model is trained on 1994 US Census data and may not reflect current socio-economic conditions.
- It is recommended to:
  - Regularly retrain the model on updated datasets.
  - Conduct fairness audits, particularly for underrepresented groups.
  - Use the model only for research or exploratory purposes unless validated for production.