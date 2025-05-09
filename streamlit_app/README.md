# Alzheimer's Diagnosis System - Troubleshooting Guide

## Fixing the Analysis History View

If you're experiencing issues with the Analysis History tab showing a blank page or displaying raw HTML code, follow these steps:

### Method 1: Run the Fix Tool (Recommended)

1. Open a command prompt in the `streamlit_app` directory
2. Run the fix tool:
   ```
   python fix_history_view.py
   ```
3. Follow the on-screen instructions:
   - Click "Fix Database Records" to repair any data issues
   - Click "Add Test Analysis Data" if needed
   - Click "Add Comparison Data" to ensure you can compare visits
   - Click "Clear Cache" to reset Streamlit's cache

4. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

### Method 2: Manual Fixes

If the fix tool doesn't solve your issue, you can try these manual steps:

1. Check for database issues:
   ```
   python comparison_fix.py
   ```

2. Add test data if needed:
   ```
   python create_test_data.py
   ```

3. Ensure there are at least two analyses for a patient in the database

4. Restart the Streamlit app and clear your browser cache

## Common Errors and Solutions

### KeyError: 'scan_path'

This error occurs in the MRI scan section. We've fixed it by adding proper checks before accessing the scan path.

### HTML Code Showing Instead of Tables

This happens when Streamlit doesn't properly render the HTML. We've fixed it by:
1. Adding inline CSS styles in the visit_comparison.py file
2. Using st.write() instead of st.markdown() for better HTML rendering
3. Adding explicit styling attributes to the HTML tables

### Missing Analyses in History

If you don't see any analyses in the history tab:
1. Make sure you've added at least one analysis for the patient
2. Check that the database connection is working properly
3. Try running the test data script to add sample analyses

## Using the Comparison Feature

1. Navigate to the "🧠 Alzheimer's Analysis" section
2. Click on the "Analysis History" tab
3. Select two different analyses using the dropdown menus
4. Click the "📊 Compare Selected Visits" button
5. You should now see a detailed comparison with tables and charts

## Need More Help?

If you continue to experience issues, please:

1. Check the Streamlit app logs for any specific error messages
2. Make sure your MySQL database is running and accessible
3. Verify that all required Python packages are installed:
   ```
 