import streamlit as st
import time
import os

# Add this import statement only if the files exist
try:
    from comparison_fix import fix_analysis_records, add_sample_comparison_data
    from create_test_data import create_test_analyses
    FIXES_AVAILABLE = True
except ImportError:
    FIXES_AVAILABLE = False

def fix_history_view():
    st.set_page_config(page_title="Fix History View", layout="centered")
    
    st.title("🛠️ Fix Alzheimer Analysis History")
    st.markdown("""
    This tool helps fix the issue where the Analysis History tab in the doctor view shows a blank page
    or displays HTML code instead of properly formatted tables.
    """)
    
    if not FIXES_AVAILABLE:
        st.error("""
        The required fix scripts are not available. Please make sure you have the following files:
        - streamlit_app/comparison_fix.py
        - streamlit_app/create_test_data.py
        """)
        return
    
    # Step 1: Database Fixes
    st.header("Step 1: Fix Database Issues")
    st.markdown("This will repair any inconsistent data in the analysis records.")
    
    if st.button("1️⃣ Fix Database Records", type="primary"):
        with st.spinner("Fixing database records..."):
            # Run the fix_analysis_records function
            fix_analysis_records()
            st.success("Database records fixed! ✅")
    
    # Step 2: Add Test Data
    st.header("Step 2: Add Test Data (If Needed)")
    st.markdown("This will add test analysis records for patients who don't have enough data for comparison.")
    
    if st.button("2️⃣ Add Test Analysis Data", type="primary"):
        with st.spinner("Adding test data..."):
            # Run the create_test_analyses function
            create_test_analyses()
            st.success("Test data added! ✅")
    
    # Step 3: Add Comparison Data
    st.header("Step 3: Ensure Comparison Data")
    st.markdown("This will make sure there's enough data for the visit comparison feature.")
    
    if st.button("3️⃣ Add Comparison Data", type="primary"):
        with st.spinner("Adding comparison data..."):
            # Run the add_sample_comparison_data function
            add_sample_comparison_data()
            st.success("Comparison data verified! ✅")
    
    # Step 4: Clear Cache
    st.header("Step 4: Clear Streamlit Cache")
    st.markdown("This will clear the Streamlit cache and session state to ensure fresh data is loaded.")
    
    if st.button("4️⃣ Clear Cache", type="primary"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Try to clear cache directory
        try:
            cache_dir = os.path.join(os.path.expanduser("~"), ".streamlit", "cache")
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                st.success("Cache cleared! ✅")
            else:
                st.info("No cache directory found.")
        except Exception as e:
            st.warning(f"Could not clear cache directory: {e}")
        
        st.success("Session state cleared! ✅")
    
    # Final instructions
    st.markdown("""
    ## Next Steps
    
    After completing the steps above:
    
    1. Restart the Streamlit app
    2. Login to the doctor view
    3. Select a patient
    4. Navigate to the 🧠 Alzheimer's Analysis section
    5. Click on the "Analysis History" tab
    6. Select two different analyses to compare
    7. Click the "📊 Compare Selected Visits" button
    
    The comparison should now display properly with formatted tables and visualizations.
    """)
    
    st.markdown("""
    ---
    ### ℹ️ Technical Details
    
    The fix addresses the following issues:
    
    - SQL query format in the doctor_view.py file
    - HTML rendering issues in the visit_comparison.py file
    - Database records with inconsistent formats
    - Missing or insufficient analysis records for comparison
    """)

if __name__ == "__main__":
    fix_history_view() 