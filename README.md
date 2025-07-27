# Domain Name Generator - Technical Report
An LLM based model for suggesting domain names for businesses based on basic descriptors. 
Author: Reggie Bain


## 1. Methodology & Initial Results
• Dataset creation approach and baseline model selection
• Initial model performance and evaluation metrics
## 2. Edge Case Analysis
• Discovery process: How you found edge cases
• Failure taxonomy: Categories of failures with examples
• Frequency analysis: How common each failure type is
## 3. Iterative Improvement
• Improvement strategies: What you tried and why
• Quantified results: Before/after metrics for each iteration
• LLM judge validation: How you ensured evaluation quality
#### Hyperparameter Search
- Due to Kaggle free tier constraints on RAM and output file sizes, we search by hand thorugh some hyperparameters.
## 4. Model Comparison & Recommendations
• Performance comparison: Statistical significance of improvements
• Production readiness: Which version you'd deploy and why
• Future improvements: Next steps for continued improvement

## 5. API Deployment
#### Tips for Use
```
make train     # fine-tune the model
make eval      # evaluate with BLEU, ROUGE, etc.
make serve     # start the FastAPI server
make test      # run edge case test script
make api_call  # test with curl
make clean     # remove cache/logs

```
