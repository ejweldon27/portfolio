# Marketing Analytics Portfolio

Two Python notebooks from my MBA coursework in Marketing and Data Science at Santa Clara University, and a notebook showing the method behind a search research tool I built for work. The notebooks start from a marketing question, not a modeling exercise — the analysis exists to support a spending or targeting decision, and each one ends with the recommendation I'd actually make.

I'm a B2B demand generation and growth marketer. These are here because I'd rather build the segmentation and attribution myself than file a ticket for it.

---

## 1. Predicting Purchase Intent from Session Behavior

**Question:** Which website sessions are likely to end in a purchase, and which behavioral signals identify them early enough to act on?

**Data:** [UCI Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) — ~12,330 e-commerce sessions with page-type counts, time-on-page, bounce and exit rates, Google Analytics page values, traffic source, visitor type, and proximity to special days. The target class is heavily imbalanced, which drives most of the modeling decisions below.

**Approach:**
- Preprocessing pipeline with `make_column_transformer` — imputation, scaling for numeric features, one-hot encoding for categoricals
- Compared a **Gradient Boosting Classifier** and a **Multi-layer Perceptron (MLP)**
- Hyperparameter tuning via `GridSearchCV`
- Evaluated on **balanced accuracy and ROC AUC** rather than raw accuracy, since a model that predicts "no purchase" every time scores well on accuracy alone and is useless

**Results:** Best model reached **0.83 accuracy, 0.83 balanced accuracy, and 0.83 ROC AUC** on held-out data. Confusion matrices, ROC curves, and precision-recall curves are included for both models.

**Why it matters for marketing:** Page value and visitor type carry most of the signal. A model like this is the basis for deciding which sessions deserve retargeting spend and which should be left alone — the same scoring logic I use on B2B accounts, applied to sessions.

> **Attribution:** team project, built with two classmates.

---

## 2. Which Marketing Channels Actually Drive Revenue (P&G)

**Question:** P&G spends across five channels. Which ones move revenue, and what is the return on each dollar?

**Data:** Course-provided dataset of P&G marketing activity — social media posts (X/WeChat), events, and in-store, digital, and TV spend, against revenue.

**Approach:**
- Exploratory analysis and correlation review per channel
- **OLS multiple regression** (`statsmodels`) to isolate each channel's contribution, with single-variable models run alongside to check stability
- Coefficients read as marginal revenue per unit of spend

**Results:** The full model explains **82% of revenue variance (R² = 0.820, adjusted R² = 0.818)**. Estimated return per dollar:

| Channel | Return per $1 |
|---|---|
| In-store | $4.00 |
| Digital | $3.31 |
| TV | $1.66 |
| Social media | No measurable effect |
| Events | Slight negative |

**Recommendation:** Shift budget toward in-store and digital. Pause event spend pending a review of why it isn't converting, and audit the social strategy with sentiment analysis before writing the channel off — low posting volume, not low channel value, may be the problem.

**Why it matters for marketing:** This is channel attribution and budget allocation, which is the same argument I make with dashboards in a live role — just with the regression shown rather than assumed.

---

## 3. Which Keywords Can a Smaller Company Win? (SXO)

**Question:** A B2B company with a small content team can't write for every keyword. Which ones are already ours, which are open, which are worth fighting a competitor for, and which aren't worth the effort?

**Data:** a built-in sample for a fictional company, Example Co, which sells inventory forecasting software, and two fictional competitors. The notebook shows the core method of a search research tool I built to plan website copy and content for a B2B IoT company. The full tool pulls the same inputs live: crawled pages, Google Autocomplete for demand, and Brave Search for rankings.

**Approach:**
- **Distinctive phrases:** field-weighted TF-IDF (a phrase in a title or URL counts 5 times, an H1 3 times, a subheading twice, body text once), so each site's positioning shows up as the phrases only it uses
- **Demand, fit and intent:** Autocomplete variants as a relative demand signal, fit as overlap with what the company sells, and searcher intent from who the top results are written for
- **Labels:** every keyword is ours, open, fight for, not winnable or not worth it, by rules on fit, intent and who ranks
- **Opportunity score:** √demand × fit × beatability × intent, so a specific phrase the company can credibly own beats a broad one it can't
- **Draft check:** a draft page is checked for the keyword in the H1 and first paragraph, an answer up front, question headings, an FAQ, numbers that trace to confirmed facts, and absolute claims a competitor could disprove

**Results:** in the sample, "reduce stockouts" ranks first: it fits exactly and no competitor ranks for it. Two keywords are worth fighting for, because competitors rank only at #5 and #9. "Warehouse management system" has high demand but scores 0, because it's not what Example Co sells. The sample draft passes the structure checks but fails three that would stop it from going live: an untraced statistic, "the only", and too few FAQ questions.

**Why it matters for marketing:** most keyword tools rank terms by search volume. This ranks them by whether a smaller company can actually win them and whether the searcher is a buyer, which is the question that decides what the content team writes first.

---

## Running the notebooks

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn scipy jupyter
jupyter notebook
```

Built on Python 3 (Anaconda). The SXO notebook needs only pandas and runs as-is, with its sample data built in. The online shoppers dataset is public and linked above. The P&G dataset was provided for coursework and is not redistributed here, so that notebook reads as a record of the analysis rather than something you can re-run as-is.

---

## About

Eric Weldon — demand generation and growth marketing, B2B IoT and hardware.
MBA, Marketing and Data Science, Santa Clara University.

[LinkedIn](https://www.linkedin.com/in/eric-weldon/)
