Customer Segmentation in Python
===============================

This is a small project I created to demonstrate how to do **customer
segmentation**. In case you don't know what it is, you can read about the
subject in [this article](https://www.forbes.com/advisor/business/customer-segmentation/).

In this project, we'll use sales data from an online store. By analyzing this
data, we can extract important information about that store's customers. In
turn, this information allows us to divide these customers into groups
(segments) that reflect their buying patterns. In the context of this project,
this is what we mean by customer segmentation.

This kind of analysis is very useful for marketing purposes. The idea is to
adopt different marketing strategies for different customer segments. For
instance, you may want to offer premium products to customers who usually spend
a lot of money. On the other hand, you may want to offer a special discount to
bring back customers who haven't purchased in a while.

Customer segmentation is a much broader topic. However, for you to understand
what we're trying to accomplish here, this quick explanation should be enough.

## Dataset

In this project, we'll consider the following dataset:

[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

This dataset is well-known. It contains a year's worth of transactional data
for a UK online retail store. For more details, please visit the link above.

## Project Files

The analysis we carry out is long. For this reason, we believe it's better to
organize this project by dividing it into several files. Specifically, this
project contains several Jupyter notebooks. Next, we briefly describe the
content of each notebook.

- Data preparation: In this notebook, we read the original dataset, and do a
  little data cleaning. After checking that the remaining data is consistent,
  we perform data aggregation. The goal is to transform transaction data into
  customer data so that later we can do the segmentation.
    * [Jupyter notebook](https://github.com/mwoitek/python-customer-segmentation/blob/master/notebooks/online_retail/1_data_preparation.ipynb)
    * [Python script](https://github.com/mwoitek/python-customer-segmentation/blob/master/notebooks/online_retail/1_data_preparation.py)
