import os
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
from chat2plot import chat2plot
import streamlit as st
from langchain_openai import AzureChatOpenAI


# Set your OpenAI API key
#os.environ["PANDASAI_API_KEY"] = 'Your API key'  # add API key

os.environ["AZURE_OPENAI_API_KEY"] = "b9135a15c242432cb20ddc43fea3a413"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-oe.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-35-turbo"


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


def preprocess_pandasai(dataframe, language_model):
    """Preprocess the DataFrame using PandasAI."""

    #create an instance of the SmartDataframe class
    smart_df = SmartDataframe(dataframe, config={"llm": language_model})
    st.write("\n\n")
    prompt = st.text_input("Describe the preprocessing task (leave empty to finish):")
    st.write("\n")
    if prompt:
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


def plot_chat2plot(dataframe, language_model):
    """Plot the DataFrame using chat2plot."""
    prompt = st.text_input("Describe the plotting task (leave empty to finish):")
    st.write("\n")
    if prompt:
        try:
            st.markdown(f":blue[Plotting task:] {prompt}")
            st.write("\n\n\n")
            c2p = chat2plot(dataframe, chat=llm)  #Creates an instance of Chat2Plot.
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

# Streamlit app
st.title("PandasAI Data Preprocessing and Chat2Plot Plotting App")
st.markdown(":green[**This app will allow you to preprocess data and plot dataframes using AI LLM libraries**]")
st.write("\n\n")

# File upload
st.subheader("PandasAI")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if df is not None:
        st.write("\n\n")
        st.markdown(":red[**Original CSV DataFrame:**]")
        st.write(df)
        
        '''creating an instance of the OpenAI llm and passing it as argument to save resounces
        and avoid instantiating an instance within the function itself everytime'''
        #llm = OpenAI(api_token=os.environ["PANDASAI_API_KEY"])
        llm = AzureChatOpenAI(openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                              azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"])
        processed_df = preprocess_pandasai(df, llm)
        
        st.write("\n\n")
        st.markdown(":red[**Processed CSV DataFrame:**]")
        st.write(processed_df)
        
        st.write("\n\n")
        st.subheader("Chat2Plot")
        plot_chat2plot(processed_df, llm)


    


