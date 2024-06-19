import os
import pandas as pd
from pandasai.smart_dataframe import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm.openai import OpenAI
from chat2plot import chat2plot
import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")[1:]
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("OPEN_AI_DEPLOYMENT_NAME")

@st.cache_data
def load_csv(file):
    """Load a CSV file into a DataFrame."""
    try:
        dataframe = pd.read_csv(file)
        st.success("The CSV file has been loaded successfully!")
        return dataframe
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
    except pd.errors.ParserError:
        st.error("The CSV file is invalid.")
    except pd.errors.DtypeWarning:
        st.error("The CSV file has mixed data types.")
    except FileNotFoundError:
        st.error("The file was not found.")
    except Exception as e:
        st.error(f"There is an error loading the CSV file: {e}")
    return None


def preprocess_pandasai(dataframe, language_model, f_key):
    """Preprocess the DataFrame using PandasAI."""

    #create an instance of the SmartDataframe class
    smart_df = SmartDatalake(dataframe, config={"llm": language_model})
    # lake = SmartDatalake([df1, df1], config={"llm": language_model})
    st.write("\n\n")
    prompt_key = f"preprocess_{f_key}"
    prompt = st.text_input("Describe the preprocessing task (leave empty to finish):", key=prompt_key)
    st.write("\n")
    if prompt != "Describe the preprocessing task (leave empty to finish):":
        try:
            st.markdown(f":blue[Processing task:] {prompt}")
            st.write("\n\n\n")
            # retrieve a dataframe variable that has been processed based on user prompt
            processed_df = smart_df.chat(prompt)
            st.success("Task has been completed successfully!")
            return processed_df
        except ValueError:
            st.error("There was a value error during processing.")
        except TypeError:
            st.error("There was a type error during processing.")
        except KeyError:
            st.error("A key error occurred during processing.")
        except Exception as e:
            st.error(f"The processing task has run into an error: {e}")
    return dataframe


def plot_chat2plot(dataframe, language_model, f_key):
    """Plot the DataFrame using chat2plot."""
    prompt_key = f"plot_{f_key}"
    prompt = st.text_input("Describe the plotting task (leave empty to finish):", key=prompt_key)
    st.write("\n")
    if prompt:
        try:
            st.markdown(f":blue[Plotting task:] {prompt}")
            st.write("\n\n\n")
            c2p = chat2plot(dataframe, chat=language_model)  #Creates an instance of Chat2Plot.
            result = c2p(prompt) #contains the final result after executing the prompt
            st.plotly_chart(result.figure, use_container_width=True)  #use Plotly package to plot
            st.success("Task has been completed successfully!")
        except ValueError:
            st.error("There was a value error during plotting.")
        except TypeError:
            st.error("There was a type error during plotting.")
        except KeyError:
            st.error("A key error occurred during plotting.")
        except Exception as e:
            st.error(f"The plotting task has run into an error: {e}")

# Streamlit app setup
st.title("PandasAI Data Preprocessing and Chat2Plot Plotting App")
st.markdown(":green[**This app allows you to preprocess and plot dataframes using AI LLM libraries**]")
st.write("\n\n")

# Multiple file upload
uploaded_files = st.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for index, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"File: {uploaded_file.name}")
        df = load_csv(uploaded_file)
        if df is not None:
            st.write(df)
            
            # LLM instance creation
            llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                openai_api_version=AZURE_OPENAI_API_VERSION,
                deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                openai_api_key=AZURE_OPENAI_API_KEY
            )

            # Preprocess and display DataFrame
            file_key = f"{uploaded_file.name}_{index}" #file_key is required to create seperate instances of widgets with unique keys
            processed_df = preprocess_pandasai(df, llm, file_key)
            if processed_df is not None:
                st.markdown(":red[**Processed DataFrame:**]")
                st.write(processed_df)

            # Plotting
            st.subheader("Plot")
            plot_chat2plot(processed_df, llm, file_key)
else:
    st.markdown(":red[**Please upload one or more CSV files.**]")