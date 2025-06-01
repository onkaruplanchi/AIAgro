import streamlit as st, base64
from Crop_Recommendation_System import app as crop_app
from Plant_Disease import app as disease_app

# Set page configuration first
st.set_page_config(page_title="Smart Agriculture Assistant", page_icon="🌾")



# Custom CSS for sidebar and navigation
def custom_css():
    st.markdown(
        """
        <style>
        /* Main Title Styling */
        .main-title {
            text-align: center;
            font-family: Open Sans;
            font-size: 45px;
            color: white;
            font-weight: bold;
            padding: 20px 0;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.3);
            background-color: #032e16;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f0f3f4;
            padding: 20px;
        }

        /* Tab Header Styling */
        div[data-baseweb="tab-list"] {
            text-align: center;
            display: flex;
            background-color: #032e16;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        

        /* Active Tab */
        div[data-baseweb="tab"] button[aria-selected="true"] {
            background-color: #4caf50;
            font-weight: bold;
            color: #ffffff !important;
            border-radius: 8px;
        }

        /* Inactive Tab */
        div[data-baseweb="tab"] button[aria-selected="false"] {
            background-color: #c8e6c9;
            color: #2e7d32;
            border-radius: 8px;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-family: 'Arial', sans-serif;
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main App Layout
def main():
    # Apply custom CSS
    custom_css()

    # App title
    st.markdown("<div class='main-title'>🌾 Smart Agriculture Assistant</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    tabs = st.tabs(["🌱 Crop Recommendation", "🩺 Plant Disease Detection"])
    with tabs[0]:
        crop_app()

    with tabs[1]:
        disease_app()

    # Optional Footer Section
    st.markdown('<div class="footer">Powered by Streamlit & AI</div>', unsafe_allow_html=True)

# Run the main app
if __name__ == "__main__":
    main()
