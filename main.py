# !/usr/bin/env python3
# coding: utf-8

import os
import streamlit as st
import streamlit.components.v1 as components
import markdown

def main():
    st.set_page_config(page_title="NLP App", page_icon=":guardsman:", layout="wide")
    st.title("NLP App with Streamlit")

    # Get a list of Markdwon files in the 'blog' directory
    markdown_files = [file for file in os.listdir("blog") if file.endswith(".md")]

    # Display a dropdown list of the markdown files
    selected_markdown_file = st.sidebar.selectbox("Select a blog post", markdown_files)

    with open(f"blog/{selected_markdown_file}", "r") as f:
        markdown_text = f.read()

    markdown_html = markdown.markdown(markdown_text)

    components.html(markdown_html, width=1200, height=800, scrolling=True)


if __name__ == "__main__":
    main()

