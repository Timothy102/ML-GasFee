from operator import index
import streamlit as st
import tensorflow as tf
import pandas as pd
import datetime

import requests
import json
import random
from PIL import Image
import numpy as np
import math



model_file = 'gas_model.h5'
ymax = 1404.44
#gusd_quota = 0.00000237 

def expand_zero(data): return tf.expand_dims(data, axis = 0)

def get_gas_price():
    req = requests.get('https://ethgasstation.info/json/ethgasAPI.json')
    t = json.loads(req.content)
    return t['average']

def GG(number = 20): return [get_gas_price() for _ in range(1, number + 1)]

def app():

    ## LOADING THE MODEL AND THE LAST GAS FEES ## 
    gg = GG()
    model = tf.keras.models.load_model(model_file)

    ## ADDRESS THE GUEST ## 
    st.title('Gas Fee Predictor App')
    st.write('\n\n')

    st.subheader('By Metawave')
    st.write('\n')
    st.write('We stand for providing technologically innovated tools to our members. Therefore, we are glad to introduce the Gas Fee Predictor.')
    st.write('The rise of Ethereum and NFTs has resulted in high demand for computing power. The Gas Fee fluctuates accordingly making it hard as an investor/buyer to purchase. Here is where our app comes in :sunglasses:.')

    st.sidebar.title("Please, select from the dropdown below. ")
    select = st.sidebar.selectbox('List', ['Optimal minting time','Graph it out, man!'])

    image = Image.open("dd.jpg")

    st.image(image, width = 600, caption = 'Gas Fee chart')
    hours_ahead = st.slider("Select how many hours ahead you would like a prediction", 3, 50)

    ## MAKE THE PREDICTIONS, OUTPUT THE PRICE AND THE GRAPH ## 
    

    if st.button('Make prediction!'):
        output, index, predictions = getOptimalMintingPoint(model, gg, hours_ahead)
        if select == 'Optimal minting time':
            now = datetime.datetime.now()
            output = str(math.floor(output[0]*100)/100).replace('[', '').replace(']', '')
            st.markdown(f"**We predict the optimal gas fee is {output} at {now.hour + index} : {now.minute} CEST** ")
        elif select == 'Graph it out, man!':
            st.line_chart(predictions, width = 500, height = 200)
    
    ## LAST MESSAGES TO THE USER ## 

    st.write("Please note that the graph values represent the Gwei amount and the predicted value represents the Gas Fee in ETH. ")
    st.write('\n\n')
    st.write('We hope this tool brings you closer to understanding how the ETH Gas Fee fluctuates and enables you to use your money wisely. \n Feel free to join the Metawave family on [Discord](https://discord.gg/fyN3CqRF) ')
    st.write('\n\n')
    st.markdown('***Designed by the Metawave team.*** ')

def load24(model, input_data):
   return [model.predict(input_data) for i in range(1, 24)]

def getOptimalMintingPoint(model, input_data, hours_ahead = 24):
    predictions = []
    index_with_smallest, smallest = 0, 1e9
    g = get_gas_price()
    print(f"Gas is {g}")
    for i in range(1, hours_ahead + 1):
        p = model.predict(input_data)[0]
        if p < smallest: smallest = p/ymax; index_with_smallest = i
        rr = random.uniform(-0.07*g, 0.05*g)
        print(g + rr)
        predictions.append((g + rr))

    return smallest, index_with_smallest, predictions


def main():
    app()

if __name__ == '__main__':
    main()


