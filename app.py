import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    st.title("Linear Regression")

    entries= st.slider(label="number of points:", min_value=1, max_value=100)
    q = st.slider(label="enter angular coefficient: ", min_value=0, max_value=100)
    d = st.slider(label="enter standard deviation:", min_value=0, max_value=100)
    n = st.slider(label="enter some noise:", min_value=1, max_value=5)

    make_random = np.random.RandomState(666)
    x = 10 * make_random.rand(entries)
    noise = np.random.normal(0, d, entries)
    y = x *q + noise *n 

    X = x.reshape(-1, 1)
    model = LinearRegression(fit_intercept=True)
    model.fit(X,y)
    y_pred = model.predict(X)

    fig = plt.figure()
    plt.scatter(x, y)
    plt.plot(X, y_pred, '-r')
    plt.axis([0, 10, 0, 200])
    st.pyplot(fig)
    

  




if __name__ == '__main__':
    main()