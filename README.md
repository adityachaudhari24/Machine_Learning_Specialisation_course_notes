# Machine Learning Specialization course - Notes
This repository contains the notes and key concepts from below resources 
1) Machine Learning Specialization course by Andrew Ng on Coursera.
2) Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron.
3) my own general research on the web and different books.

# Module 1: Supervised Machine Learning: Regression and Classification

<details>
<summary>🎯Q. What is machine learning?</summary>

- Machine Learning is the `science (and art) of programming computers` so they can learn from data.
- Machine learning is the field that gives abilities to computers to learn without being explicitly programmed. - Arthur Samuel (1959)
- ⭐⭐Machine learning⭐ and ⭐deep learning⭐ are fields aimed at `implementing algorithms` that enable computers to learn from data and perform tasks that typically require human intelligence.⭐
- There are below main types of machine learning:
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Each type has its own unique characteristics and applications.
</details>


<details>
<summary>🎯Q. What is supervised learning in ML? Different types (3 key types)</summary>

- `supervised learning` is an algorithm which maps input X to output Y, where the learning algorithm learns from the "right answers" (labeled data).
- `Supervised learning` is a type of machine learning where the model is trained on a labeled dataset, meaning that each training example is paired with an output label.

- `Two main types of Supervised Learning` - 
  - `Regression` - Predicting continuous values (e.g., house prices, stock prices).
  - `Classification` - Predicting discrete labels or categories (e.g., spam detection, image recognition).

⭐⭐ Regression supervised learnning ⭐⭐
- `Regression meaning` - Its finding the relationship which explains how one element depends on another. 
  - `Regression` also means trying to `predict a number` from infinitely `possible outputs`.
  - `Example` - Predicting house prices based on features like size, location, number of rooms, etc.
  - in ecommerce , given input features of user, product, context, predict purchase amount.


⭐⭐ Classification supervised learning ⭐⭐
- `Classification meaning` - Its finding the relationship which explains how one element belongs to a particular category or class.
  - `Classification` also means trying to `predict a label` from a `finite set of possible outputs`.
  - `Example` - Email spam detection (spam or not spam), image recognition (cat, dog, car, etc.)
    - in ecommerce , given input features of user, product, context, predict whether user will buy the product (yes/no)



- IMP One liners - 
- 💡 Its not always finding the straight line in supervised learning regression; this can be polynomial, logarithmic, exponential, etc.
- Classification algorithms predicts categories / classes / labels unlike regression which predicts continuous values( e.g., real numbers).
</details>


<details>
<summary>🎯Q. What is unsupervised learning in ML?</summary>

- In `unsupervised learning` we ask algorithms to find something intresting in unlabeled data. Where in `supervised learning` the algorithm learns from the labeled data.
- Examples 
  - `Anomaly detection` - identifying unusual data points that do not conform to expected behavior.
  - `Clustering` - grouping similar data points together based on their features.
  - `Dimensionality reduction` - compress data using fewer numbers, reducing the number of features in a dataset while preserving important information.

</details>

<details>
<summary>🎯Q. What is linear regression model?</summary>

- ![alt text](image.png)
- For linear regression, the model is represented by:

  **f<sub>w,b</sub>(x)<sup>(i)</sup> = wx<sup>(i)</sup> + b**

  Where:
  - **f<sub>w,b</sub>(x)** is the prediction function
  - `w` is the weight/slope parameter
  - `b` is the bias/y-intercept parameter
  - `x` is the input feature
- The formula above is how you can represent straight lines - different values of  𝑤 and  𝑏 give you different straight lines on the plot.
- The formula can be used to predict outcomes based on input features, making it a fundamental concept in machine learning.
- ![alt text](image-1.png)
- ![alt text](image-2.png)
- ![alt text](image-3.png)
</details>

<details>
<summary>🎯Q. what is cost function ?</summary>

- The `cost function` (or `loss function`) is used to define the best parameters for the model, in our liniear regression case the best values for w and b.
- The goal of linear regression is to find the parameters `w` or `w` and `b` that results in smallest possible value for the cost J.
- ![alt text](image-4.png)
- ![alt text](image-5.png)
- ![alt text](image-6.png)
- ![alt text](image-7.png)
- ![alt text](image-8.png)
- ![alt text](image-9.png)
- ![alt text](image-10.png)
</details>

<br>
<br>
<br>
********************************* Ignore below *********************************
<details>
<summary>Emojis used</summary>
⭐ - For important points
🔥 - super important
💡 - For key concepts/tips
⚠️ - For warnings/common mistake
🎯 - For exam targets/focus areas/ question 
🚀 - For advanced topics .
🚫 - For indicating something that cannot be used or a concerning point
</summary>
</details>

<details>
<summary>🎯Q. fsdfsdf</summary>

- 

</details>