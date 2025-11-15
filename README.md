# Boosting-to-Bagging-Parametric-Boosting-Tree-Cohorts-and-RF-Tuning
**Overview**   
	•	A hands-on, end-to-end study across stage-wise boosting, L1 methods, decision trees, bagging, and random forests.   
	•	Implements textbook exercises with clean, reproducible Python (no external data files needed).   
	•	Emphasis on coordinate-wise parametric boosting, feature split diagnostics, and OOB-based model selection.    

What’s inside 

	• Gradient Boosting for Logistic Regression (parametric learners)    
	
		•	Forward stage-wise coordinate descent with 1-D Newton line-search (logistic loss).   
     	•	Two modes: without and with intercept as a search direction.   
    	•	Outputs: loss paths, coordinate selection trajectories, GLM comparison.
			
	• Churn Classification (Telco via OpenML)
	
	  	•	Reuses the 12.8 booster inside a sklearn preprocessing pipeline (impute + one-hot).
	  	•	Metrics: accuracy, precision, recall, F1 (+ optional Average Precision one-liner: average_precision_score(y_true, y_score)).
	  	•	Result snapshot: Acc 0.798, Prec 0.647, Recall 0.527, F1 0.581 (before threshold tuning).
		
	• L2Boost ν-sweep (learning-rate vs selection frequency)
	
	  	•	Synthetic correlated design; compare ν ∈ {1.0, 0.5, 0.1, 0.01}.
	  	•	Plots: train MSE paths, feature selection frequencies per ν.
	  	•	Takeaway: small ν → smoother path, more evenly shared selections among correlated predictors.
	  	•	CD-Lasso vs LARS-Lasso
	  	•	Simulation framework comparing test MSE, support F1, and runtime across ρ and p.
	  	•	Notes on sklearn API changes handled (e.g., normalize removal for LARS).
		
	•	Shallow Tree Cohorts with Odds Ratios (Breast Cancer)
	
		•	Fit a ~6-leaf tree; compute leaf-level malignant odds and relative odds vs baseline.
		•	Clear rules (e.g., worst radius > 16.8 & mean concavity > 0.072) define very-high-risk cohorts.
		•	15.3 — Gini Split Statistic Scans (Fig 15.5 style)
		•	For each feature, scan all thresholds and plot Gini gain vs threshold with permuted-label baseline.
		•	Top variables: worst radius/area/perimeter and (worst/mean) concave points/concavity (peaks ≈ 0.31–0.33).
		•	Provides interpretable cutpoint candidates.
	
	•	k-NN Consistency (proof sketch)
		•	Shows k-NN as a local average satisfying conditions under k→∞, k/n→0.
	•	Bagging vs Single Tree (Breast Cancer)
		•	Single CART vs Bagging(200).
		•	Test results (your run):
		•	Tree: Acc 0.958, Prec 0.980, Rec 0.906, F1 0.941
		•	Bagging: Acc 0.965, Prec 1.000, Rec 0.906, F1 0.951, OOB Acc 0.955
		•	Interpretation: bagging removes false positives (↑ precision) while recall stays similar.
	• Majority Vote Under Independence
		•	Derives p_B = \Pr(K \ge \lceil B/2\rceil) with K\sim \text{Bin}(B,p); explains why correlation breaks the “perfect as B→∞” argument.
	•	RF Regression: nodesize (min_samples_leaf) Tuning
		•	Friedman-1 simulation; sweep nodesize and select by OOB MSE vs test MSE.
		•	In the high-signal, large-n setting: nodesize=1 wins for both; OOB closely tracks test error.
