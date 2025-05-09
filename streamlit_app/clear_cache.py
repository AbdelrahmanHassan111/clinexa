import streamlit as st
import os
import shutil

def clear_streamlit_cache():
    """Clear Streamlit cache to fix display issues"""
    try:
        # Display information
        st.title("Cache Cleaner Utility")
        st.write("This tool helps fix issues with the Alzheimer Analysis History view.")
        
        if st.button("🗑️ Clear Streamlit Cache", type="primary"):
            # Clear Streamlit cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".streamlit", "cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                st.success("✅ Cache cleared successfully!")
            else:
                st.info("No cache directory found.")
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("✅ Session state cleared!")
            st.info("Please restart the app and try viewing the Analysis History again.")
    
    except Exception as e:
        st.error(f"Error clearing cache: {e}")

if __name__ == "__main__":
    clear_streamlit_cache() 