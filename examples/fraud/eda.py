# ruff: noqa
#!/usr/bin/env python

# # BNP Fraudsters - EDA
#
# ## 1. Introduction
#
# This challenge is a use-case from by BNP Paribas Personal Finance, hosted on the platform [challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/104/). The dataset is under a [public and permissive license](https://etalab.gouv.fr/licence-ouverte-open-licence/).
#
# The bank offers credit to customers of retailers and web merchants and is exposed to various frauds.
#
# > The Credit Process Optimization team supports local risk teams to improve credit processes' efficiency, balance profitability, customer journey, and risk profiles.
#
# Although mentioned, the context in which the models are used is not precise enough to derive operational constraints and a precise utility function to optimize. Instead, the challenge suggests maximizing the average precision (i.e. the PR curve AUC). We will stick, however, to a virtual utility function using our best guess.
#
# Finally, the features available are rather limited, e.g. we don't have access to customer data, merchant data, or any temporal information. We only have access to the products in the customer basket (up to 24), with the following attributes for each product:
# - `item` (category of the product)
# - `make` (brand of the product)
# - `model` (the name of the model of the product)
# - `goods_code` (retailer code, we don't have a precise definition of this variable, we assume it is the product reference across different shops)
# - `Nbr_of_prod_purchas` (the number of products selected)
# - `cash_price` (the individual price of a single product)
#
# Since most consumers only select a few different products when applying for a loan, the features are sparse beyond the first two selected products, making fraud detection even more challenging.
#
# ## 2. Exploring the variables
#
# The challenge organizers have assembled the data in a single table, which eases our work. Let's load this file:

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from mandr.examples.fraud.eda_plots import get_group_cols
from matplotlib import pyplot as plt

X = pd.read_csv("X_train.csv", low_memory=False).convert_dtypes()

# Rename the 24 columns 'Nbr_of_prod_purchas' in a shorter 'nbr'
renaming = dict(
    zip(
        get_group_cols("Nbr_of_prod_purchas"),
        get_group_cols("nbr"),
    )
)
X = X.rename(columns=renaming)

# Display the first item attributes
attribute_cols = ["item", "cash_price", "make", "model", "goods_code", "nbr"]
cols = ["ID", *[f"{col}1" for col in attribute_cols], "Nb_of_items"]
X[cols]


# We see products like electronics or home furnitures, with a price range close to 1000€. This price range is expected since these buyers used a credit.
#
# We also see that "item" categories have a semantic (and syntaxic) similarity. For instance, "computers" and "computer peripherals accessories" should be closer in some latent feature space than "computers" and "bedroom furniture".
#
# The same observation goes for "model", where models mentioning "MACBOOK" should be closer together. There is somewhat of a redundancy between "model" and "make", since the name of the maker is often included in the model.
#
# "Goods code" is harder to decipher, and may not carry a meaningful encoding. We will explore that later in this analysis.
#
# We can observe that buyers tend to buy only a single of the first, expensive item. This is also expected in a B2C market, where consumers don't stack TVs in their homes.
#
# Note that `Nb_of_items` is the number of **unique** items in the basket, while `nbr{i}` is the **number of instances** of the ith item.
#
# Last but not least, we can intuite that some of our features are hierarchical:

# In[2]:


import graphviz as gr

g = gr.Digraph()
g.edge("item", "model")
g.edge("make", "model")
g.edge("model", "goods_code")
g.edge("goods_code", "ID")
g.edge("cash_price", "ID")
g.edge("nbr", "ID")
g


# Let's check that our IDs are unique:

# In[3]:


X["ID"].nunique() == X.shape[0]


# Then, what is the cardinality of our features? We only select the first item for simplicity.

# In[4]:


cols = ["ID", *[f"{col}1" for col in attribute_cols]]
X[cols].nunique().sort_values()


# It might be worth exploring the link between the risk of fraud and outliers in `nbr` like buying 16 instances of a given product.
#
# We see that the cardinality conceptually follows our hierarchical structure: `item` < `make` < `model` < `goods_code` < `ID`.
#
# Let's go further and describe the relationship between `model` and `goods_code`. How many different models are attributed to each `goods_code`?

# In[5]:


model_per_code = X.groupby("goods_code1")["model1"].nunique()
model_per_code.value_counts(normalize=True)


# While most codes are associated to a single model, 17% of codes link to 2 distinct models. This is intriguing and we count the tuples (`goods_code`, `model`) to better understand the relationships between these two attributes:

# In[6]:


duplicate_models = model_per_code[model_per_code == 2].index
(
    X.loc[X["goods_code1"].isin(duplicate_models)]
    .groupby(["goods_code1", "model1"])["ID"]
    .count()
)


# Two observations can be made:
# 1. "Retailer" looks like a placeholder, and sometimes yield two distinct models for the same code.
# 2. Duplicate models seem to be due to data entry issues otherwise.
#
# So, we can still assume conceptually a 1:1 relationship between codes and models.
#
# Let's reproduce this experiment with the pairs `item` - `model` and `make` - `item`.

# In[7]:


item_per_model = X.groupby("model1")["item1"].nunique()
item_per_model.value_counts()


# Here again, some models are linked to two distinct items. Let's display those:

# In[8]:


duplicate_items = item_per_model[item_per_model == 2].index
(
    X.loc[X["model1"].isin(duplicate_items)].groupby(["model1", "item1"])["ID"].count()
).head(6)


# The different items are highly similar, so we can also assume that the 1:1 relationship holds. Note that during modelling, we should use category transformers that account for syntaxic similarity, instead of transformers that consider all categories orthogonal.

# Finally, let's count the occurrence of distinct `make` for a given `model`.

# In[9]:


make_by_model = X.groupby("model1")["make1"].nunique()
make_by_model.value_counts()


# In[10]:


duplicate_make = make_by_model[make_by_model != 1].index
(X.loc[X["model1"].isin(duplicate_make)].groupby(["model1", "make1"])["ID"].count())


# It seems that `MADE TO MEASURE CURTAINS` is a model placeholder like `RETAILER`. So we can assume a 1:1 relationship between `model` and `make` here too.
#
# In conclusion, our initial concept schema looks valid.
#
# Before moving on to analyse the target, let's get a better understanding of the `RETAILER` placeholder. We display the first 3 items, models and makers for some lines that contains `RETAILER` somewhere.

# In[11]:


cols = []
for col_group in ["item", "model", "make"]:
    cols += get_group_cols(col_group, 3)
X.loc[(X == "RETAILER").any(axis=1)][cols].head(10)


# So, `RETAILER` is a umbrella placeholder, that cover different concepts.
# - When purchasing computers or tech gears, the store can offer services, warranties or add a fulfilment charge. These additional items are placed **at the end of the basket**, and since they are probably provided by the store, they are marked `RETAILER` by the data science team from BNP Paribas.
# - Some models of furniture (living & dining or bedroom) show `RETAILER` as a part of their name, and sometimes their entire name. This could means that they are designed by the store, rather than a specific brand.
#
# This again hints at the fact that during modelling we should indicate when `RETAILER` belongs to a model name, or use encoders that will identify close syntaxic similaries.

# ## 3. Exploring the target
#
# We now switch gears and load the target, before merging it with the feature table.

# In[12]:


y = pd.read_csv("Y_train.csv", index_col="index")
y


# As a safety measure and to mitigate the risk of overfitting, we only analyse the training data.
#
# By its nature, fraud is a very imbalanced classification problem. Here, the prevalence is close to 1.4%. Therefore, we use stratification strategies to avoid selecting too few positive classes in our analysis, and later our modelling.

# In[13]:


from sklearn.model_selection import train_test_split

target_col = "fraud_flag"
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y[target_col],
)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# We merge our training dataset with our training target to derive more insights.

# In[14]:


df = X_train.merge(y_train, on="ID")
df.shape


# As we already know from the challenge description, our target distribution is very imbalanced. We show the target distribution:

# In[15]:


palette = sns.color_palette("colorblind", n_colors=2)
ax = df[target_col].value_counts().plot.bar(color=palette)
ax.bar_label(ax.containers[0])


# In[16]:


df[target_col].value_counts(normalize=True)


# After our stratification split, the prevalance is still 1.4%. One thousand positive class is fairly low, but should be enough to ensure some predictive power.
#
# Next, we want to make sure that `Nb_of_items` is almost 100% correlated with the number of items that aren't None in each basket, for all attributes.

# In[35]:


correlations = []
for col in attribute_cols:
    cols = get_group_cols(col)
    n_not_nulls = df[cols].notnull().sum(axis=1)
    correlations.append(np.corrcoef(n_not_nulls, df["Nb_of_items"])[0][1])
ax = sns.scatterplot(x=attribute_cols, y=correlations)
ax.set_ylim([0, 1])


# That assumption looks correct.
#
# Previously, we wondered if there were a connection between the number of unique items in the basket and the fraud likelihood. Let's display the normalized distribution of unique items in non-fraudulent vs fraudulent basket. Normalized distributions help us compare two groups with a widely different sample size.

# In[18]:


fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
for idx, ax in enumerate(axes):
    sns.histplot(
        df.loc[df[target_col] == idx]["Nb_of_items"],
        log_scale=[True, False],
        bins=24,
        binwidth=0.1,
        stat="probability",
        color=palette[idx],
        ax=ax,
    )
    ax.set_title(f"{target_col}: {idx}")
    ax.grid()


# It's quite clear that extremely few baskets have more than 5 different items in them. It also seems that there is a higher proportion of basket with 2 items in the fraud case.
#
# **Are there specific categories of items for which fraudsters buy a greater number of unique products?**
#
# The suplots below loop over the 10 item categories that are the most frequently found in a fraudulent basket.
#
# For a given item category, the bars height represent the normalized distribution of the number of unique models in a basket. As most baskets only have a single item, the probability of having distinct models is close to 0, and hard to represent in the graph. To enhance readability and compare the fraudulent vs non-fraudulent basket group, we remove from the plots the cases where there is only a single item. The number above the bar gives the absolute number of fraud for this number of unique items, and the totals in the legend count the number of basket with at least one item.
#
# Here is an example on how to read these charts:
# - The first plot represent the distribution of the number of distinct computers in a basket. There is a total of 43.7k non-fraudulent and 976 fraudulent baskets with at least an item being a computer.
# - Among these, more than 4% of the fraudulent baskets have 2 distinct computers (which amount to 55 baskets), against less than 1% of non-fraudulent baskets (which amount to around less than 436 baskets).
# - This suggests that it is 4x time more frequent for fraudulent baskets to have 2 distinct computers than it is for non-fraudulent basket. By defining $C$ the random variable counting the number of unique items, we get: $$\mathbb{E}[Y] < \mathbb{E}[Y| C_{COMPUTERS} = 2]$$

# In[19]:


from mandr.examples.fraud.eda_plots import plot_n_items_per_basket

plot_n_items_per_basket(df, n_most_freq=10)


# We can notice that fulfilment charge is never ordered twice. Apart from computers, we see a higher likelihood for buying multiple distincts products for items like home cinema or audio accessories, although both represent rare events.
#
# This raises a related question: do fraudsters buy more instances of the same article? E.g. several pairs of Apple airpods or more than one computer of the same model? We use the same previous graph methodology:

# In[20]:


from mandr.examples.fraud.eda_plots import plot_nbr_per_item

plot_nbr_per_item(df, n_most_freq=10)


# Fraudsters don't stand out on buying the same item multiple times, so we can expect the `nbr{i}` feature to play little role in the fraud prediction performance.
#
# So in conclusion, a frauders' trick could be to buy different computer models instead of the same model multiple times.

# ## 4. Price distribution
#
# As fraudsters take a significant risk by not repaying their loan, we can expect their basket value to be higher than non-fraudulent baskets, and therefore we can expect price to be a important feature for prediction. Let's start by computing the total basket price.

# In[21]:


def compute_total_price(df):
    price_cols = get_group_cols("cash_price")
    nbr_cols = get_group_cols("nbr")
    df["total_price_"] = 0
    for price_col, nbr_col in zip(price_cols, nbr_cols):
        df["total_price_"] += df[price_col].fillna(0) * df[nbr_col].fillna(0)
    return df


df = compute_total_price(df)


# Then, we can explore the `total_price_` distribution of both the non-fraudulent and fraudulent baskets.

# In[22]:


from mandr.examples.fraud.eda_plots import plot_price_distribution

plot_price_distribution(df)


# In[23]:


df.query(f"{target_col} == 0")["total_price_"].describe()


# In[24]:


df.query(f"{target_col} == 1")["total_price_"].describe()


# Interestingly, the fraudster maximum outlier is far lower than the maximum legit basket. Also, for both groups the median is lower than the means, indicating a skew toward the large values.
#
# The difference in means between the two groups looks statistically significant. Let’s estimate a very simple model. We will regress the log price of baskets on the fraud indicator. We use logs here so that [our parameter estimates have a percentage interpretation](https://stats.stackexchange.com/questions/244199/why-is-it-that-natural-log-changes-are-percentage-changes-what-is-about-logs-th). With it, we will be able to say that fraud is associated to a price increase of x%.

# In[25]:


import statsmodels.formula.api as smf

result = smf.ols(f"np.log(total_price_) ~ {target_col}", data=df).fit()
result.summary().tables[1]


# So, fraud is associated with an average increase of the total basket price by 27%. Note that this relationship is not causal, since the fraud is not the treatment but the dependent variable, and the basket price is not a factor leading to fraud per se.

# We now display the price distribution for the top 10 item categories that are the most subject to fraud. We aim to find specific categories where the mean difference is higher than the average.

# In[26]:


from mandr.examples.fraud.eda_plots import plot_multiple_price_dist

plot_multiple_price_dist(df, n_most_freq=10)


# Computers, phones, audio accessory and home cinema show a difference in means higher than the average! We can expect the price for these categories to play a bigger role in the fraud prediction.

# ## 5. Co-occurrence graph
#
# We now study the co-occurrences of items and models. In this problem, baskets are essentially lists of items, of variable size, that have been flattened into a dataframe of fixed dimension. To identify what items are often bought together, we want to shift this dataframe representation to a graph that will explicit the joint frequency of having 2 specific items together in the same basket.
#
# The idea is that an item that is frequent across baskets should be represented by a node of large size, and frequent pairs of items should be represented by an edge with a large width.
#
# We noticed some data entry issues in the past, so we write a quick cleaner function to remove some typos. This will help us group items together.

# In[27]:


def clean_items(df):
    """Small cleaning step."""
    for item_col in get_group_cols("item"):
        df[item_col] = df[item_col].str.replace("& ", "").str.replace(",", "")
    return df


df = clean_items(df)


# We begin with plotting the co-occurrence graph of the legit credits and limit this representation to the `top` most frequent pairs only. Increasing this variable will result in a finer but more crowded graph.

# In[28]:


from mandr.examples.fraud.eda_plots import plot_graph

plot_graph(
    df.loc[df[target_col] == 0],
    column="item",
    top=100,
    figsize=(25, 25),
)


# We notice a few things.
# 1. `Computers` is the most represented category by far. It is frequently associated with `fulfillment charge`, and `computers accessories`.
# 2. As expected, `fulfillment charge` is a very central node and co-occurs with lots of other items
# 3. We distinguish some clusters: computers, furniture, and baby.
#
# We can now focus on the co-occurrence graph of items in fraudulent basket and try to identify differences.

# In[29]:


plot_graph(
    df.loc[df[target_col] == 1],
    column="item",
    top=100,
    node_color="coral",
    figsize=(25, 25),
)


# We notice the following:
# - `Computers` and `fulfillment charge` are still highly represented, but `computers accessories`, `warranty` and `service` are more rare.
# - `Warranty` in particular, is never associated with `television` and `computers accessories`. So, we can infer that it might be a good indicator of nonfraudulent behavior.
# - The `furniture` cluster has been replaced with `kitchen accessories`, which is quite dense. This means that these items are more prone to fraud, and are frequently bought together.
#
# In addition, we can perform the same analysis for brands instead of items, and try to identify more fraudulent patterns. We begin with nonfraudulent baskets. Note that, because the cardinality of `make` is higher than `item`, we have more long-tail/rare relationships to represent. We exclude `RETAILER` from the visualization because this node is so connected and central that it hides all the less frequent co-occurrences.

# In[30]:


plot_graph(
    df.loc[(df[target_col] == 0)],
    column="make",
    top=100,
    figsize=(25, 25),
    exclude="RETAILER",
)


# We can map some frequent items we saw previously to brands.
# - `Apple` is behind most `computers`, `phone`, and `computer accessories` items.
# - `Anyday retailer` is a central node, well connected to non-high-tech gears.
# - `Tommee tipee` is also a central node to a lesser extent.
#
# Let's now compare this to the brands from fraudulent baskets.

# In[31]:


plot_graph(
    df.loc[df[target_col] == 1],
    column="make",
    top=100,
    node_color="coral",
    figsize=(25, 25),
    exclude="RETAILER",
)


# - We see fewer interactions, which makes sense because the dataset is imbalanced.
# - `Anyday retailer` is not a central node anymore and is less purchased by fraudsters
# - We find our previous dense clusters: kitchen (Le Creuset, Skandinavisk) and baby (Chicco, 4MOMS)
# - Apple is not connected to other informatics brands like LG or Microsoft.

# ## 6. Marginal fraud ratios
#
# We now explore the marginal fraud likelihood for each item. We aim at finding the categories with a higher likelihood than the average.
#
# We compute the likelihood as: $$L_{item} = \frac{n_{item}}{N_{item}}$$ where $n_{item}$ is the number of fraudulent basket containing the item, and $N_{item}$ is the total number of baskets containing the item.
#
# - The upper x-axis represents the likelihood $L_{item}$ and is associated with the orange line. The items are sorted in decreasing order of likelihood, and the red dashed line represents the global average likelihood of 1.4%.
# - The bottom x-axis represents the absolute number of frauds $n_{item}$, and is associated with the blue bars.

# In[32]:


from mandr.examples.fraud.eda_plots import plot_fraud_ratio

plot_fraud_ratio(df, "item")


# Items above the red dashed line have a higher fraud likelihood than the dataset average. Among them:
# - The `Luggage` item category has a likelihood higher than 8%, but with only one fraudulent basket. With such a small amount of fraud, the high likelihood of luggage is probably due to chance, and the total number of baskets with `luggage` is small.
# - `audio accessories` are 3x more likely to be linked to fraud than the average (their likelihood is almost 5%). With 72 fraudulent baskets linked to that item, this category is a stronger indicator than `luggage`.
# - `Computers` have the highest number of fraud and are slightly above the average (2.2%).
