import streamlit as st
import pandas as pd
import numpy as np


def main(): 
  st.title("SCA Climate")
  num1 = st.input_number("Kindly enter your age")
  num2 = st.input_number("Enter your height")
  st.write(num1, num2)
  
  
if __name__ == "__main__":
  main()

