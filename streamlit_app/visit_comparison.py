import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px

# Try to import custom modules, with fallback for missing modules
try:
    from comparison_styles import get_comparison_styles
    from comparison_charts import (
        create_change_bar_chart, 
        create_radar_chart, 
        create_brain_measurement_chart,
        create_timeline_chart
    )
    USE_CUSTOM_MODULES = True
except ImportError:
    USE_CUSTOM_MODULES = False
    # Define fallback CSS if module not found
    def get_comparison_styles():
        return """
        <style>
            .highlight-improvement { background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
            .highlight-decline { background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
            .highlight-same { background-color: #e2e3e5; color: #383d41; padding: 2px 5px; border-radius: 3px; }
        </style>
        """
    
    # Define fallback chart functions
    def create_change_bar_chart(compared_features, feature_descriptions, limit=8):
        """Simple fallback chart function when the chart module isn't available"""
        # Simply display the data as a table instead
        top_changes = sorted(compared_features, key=lambda x: abs(x['pct_change']), reverse=True)[:limit]
        df = pd.DataFrame(top_changes)
        df['feature_name'] = df['feature'].apply(
            lambda x: f"{x}: {feature_descriptions.get(x, '').split('-')[0] if feature_descriptions.get(x, '') else ''}"
        )
        df['direction'] = df['is_decline'].apply(lambda x: "⬇️ Decline" if x else "⬆️ Improvement")
        return df[['feature_name', 'val1', 'val2', 'pct_change', 'direction']]
    
    def create_radar_chart(df, feature_category, decline_if_higher):
        """Fallback for radar chart - returns dataframe instead"""
        return df
        
    def create_brain_measurement_chart(df):
        """Fallback for brain measurement chart - returns dataframe instead"""
        return df
        
    def create_timeline_chart(visit_history, current_visits):
        """Fallback for timeline chart - returns None instead"""
        return None

# Function to compare two visits
def display_visit_comparison(comparison_data, patient_info, model, feature_descriptions):
    """
    Display a comprehensive comparison between two patient visits with highlighting of changes
    
    Args:
        comparison_data: List of two dictionaries containing visit data
        patient_info: Dictionary with patient information
        model: The generative AI model to use for insights
        feature_descriptions: Dictionary mapping feature names to descriptions
    """
    if len(comparison_data) != 2:
        st.warning("Need exactly two visits to compare")
        return
        
    # Make sure the first record is the earlier one for consistent display
    comparison_data.sort(key=lambda x: x['analyzed_at'])
    
    # Inline CSS styles instead of relying on external CSS
    st.markdown("""
    <style>
        /* Visit header styling */
        .visit-header {
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px;
        }
        
        /* Changes card styling */
        .changes-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .changes-card-header {
            background-color: #f1f5f9;
            padding: 10px 15px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .changes-card-body {
            padding: 15px;
        }
        
        .change-stat {
            text-align: center;
            padding: 5px;
        }
        
        .change-stat-value {
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .change-stat-label {
            font-size: 0.9rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Table styling */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .comparison-table th, .comparison-table td {
            padding: 10px;
            text-align: left;
        }
        
        .comparison-table th {
            background-color: #f1f5f9;
            font-weight: 600;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .comparison-table tr {
            border-bottom: 1px solid #e5e7eb;
        }
        
        /* Status indicators */
        .highlight-improvement {
            background-color: #d4edda;
            color: #155724;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .highlight-decline {
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .highlight-same {
            background-color: #e2e3e5;
            color: #383d41;
            padding: 2px 8px;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add CSS styles
    st.markdown(get_comparison_styles(), unsafe_allow_html=True)
    
    # Extract features from both visits
    visit1_data = comparison_data[0]
    visit2_data = comparison_data[1]
    
    if 'input_features' not in visit1_data or 'input_features' not in visit2_data:
        st.warning("Missing feature data for one or both visits.")
        return
        
    try:
        # Parse features from both visits
        visit1_features = json.loads(visit1_data['input_features']) if visit1_data['input_features'] else {}
        visit2_features = json.loads(visit2_data['input_features']) if visit2_data['input_features'] else {}
        
        # Format visit dates and info
        visit1_date = visit1_data['analyzed_at'].strftime('%Y-%m-%d %H:%M')
        visit1_pred = visit1_data['prediction']
        visit1_conf = float(visit1_data['confidence_score']) * 100
        
        visit2_date = visit2_data['analyzed_at'].strftime('%Y-%m-%d %H:%M')
        visit2_pred = visit2_data['prediction']
        visit2_conf = float(visit2_data['confidence_score']) * 100
        
        # Calculate days between visits
        days_between = (visit2_data['analyzed_at'] - visit1_data['analyzed_at']).days
        
        # Create enhanced header with patient info and visit comparison
        st.markdown(f"""
        <div class="visit-header">
            <h2 style="margin-top:0;">Visit Comparison Analysis</h2>
            <p style="font-size:1.1rem;">Patient: <strong>{patient_info['full_name']}</strong> | 
               Age: {datetime.now().year - patient_info['birth_date'].year} | 
               Gender: {patient_info['gender']}</p>
            <div style="display:flex; justify-content:space-between; margin-top:15px;">
                <div style="text-align:center; flex:1;">
                    <h4>First Visit: {visit1_date}</h4>
                    <p>Diagnosis: <strong>{visit1_pred}</strong> ({visit1_conf:.1f}%)</p>
                </div>
                <div style="text-align:center; width:100px;">
                    <div style="font-size:2rem; color:#6c757d;">➔</div>
                    <div style="color:#6c757d;">{days_between} days</div>
                </div>
                <div style="text-align:center; flex:1;">
                    <h4>Second Visit: {visit2_date}</h4>
                    <p>Diagnosis: <strong>{visit2_pred}</strong> ({visit2_conf:.1f}%)</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate overall change summary
        all_features = set(visit1_features.keys()) | set(visit2_features.keys())
        compared_features = []
        
        # Define features where higher values indicate decline
        decline_if_higher = ["CDRSB", "ADAS11", "ADAS13", "FAQ", "Ventricles", "RAVLT_forgetting", "RAVLT_perc_forgetting"]
        
        for feature in all_features:
            if feature in visit1_features and feature in visit2_features:
                val1 = float(visit1_features[feature])
                val2 = float(visit2_features[feature])
                pct_change = ((val2 - val1) / val1) * 100 if val1 != 0 else 0
                
                # Determine if change is improvement or decline based on feature
                is_decline = False
                if feature in decline_if_higher:
                    is_decline = val2 > val1
                else:
                    is_decline = val2 < val1
                
                if abs(pct_change) > 1:  # Only consider significant changes
                    compared_features.append({
                        'feature': feature,
                        'val1': val1,
                        'val2': val2,
                        'change': val2 - val1,
                        'pct_change': pct_change,
                        'is_decline': is_decline
                    })
        
        # Count improvements and declines
        improvements = [f for f in compared_features if not f['is_decline']]
        declines = [f for f in compared_features if f['is_decline']]
        
        # Create summary statistics section
        st.markdown("""
        <div class="changes-card">
            <div class="changes-card-header">
                Summary of Changes
            </div>
            <div class="changes-card-body">
                <div style="display:flex; justify-content:space-between;">
                    <div class="change-stat">
                        <div class="change-stat-value" style="color:#28a745;">{}</div>
                        <div class="change-stat-label">Improvements</div>
                    </div>
                    <div class="change-stat">
                        <div class="change-stat-value" style="color:#dc3545;">{}</div>
                        <div class="change-stat-label">Declines</div>
                    </div>
                    <div class="change-stat">
                        <div class="change-stat-value">{}</div>
                        <div class="change-stat-label">Total Changes</div>
                    </div>
                    <div class="change-stat">
                        <div class="change-stat-value">{}</div>
                        <div class="change-stat-label">Days Between</div>
                    </div>
                </div>
            </div>
        </div>
        """.format(
            len(improvements),  # Number of improvements
            len(declines),      # Number of declines
            len(compared_features),  # Total changes
            days_between        # Days between visits
        ), unsafe_allow_html=True)
        
        # Diagnosis change notice
        if visit1_pred != visit2_pred:
            if (visit1_pred == "Nondemented" and visit2_pred == "Converted") or \
               (visit1_pred == "Converted" and visit2_pred == "Demented") or \
               (visit1_pred == "Nondemented" and visit2_pred == "Demented"):
                st.markdown("""
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; 
                            margin: 20px 0; border-left: 5px solid #dc3545;">
                    <h4 style="margin-top:0;">⚠️ Diagnostic Status Declined</h4>
                    <p>Patient's diagnostic status has worsened since the previous visit. 
                    Consider reviewing treatment plan and intervention strategies.</p>
                </div>
                """, unsafe_allow_html=True)
            elif (visit1_pred == "Demented" and visit2_pred == "Converted") or \
                 (visit1_pred == "Converted" and visit2_pred == "Nondemented") or \
                 (visit1_pred == "Demented" and visit2_pred == "Nondemented"):
                st.markdown("""
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; 
                            margin: 20px 0; border-left: 5px solid #28a745;">
                    <h4 style="margin-top:0;">✅ Diagnostic Status Improved</h4>
                    <p>Patient's diagnostic status has improved since the previous visit. 
                    Current treatment approach appears to be effective.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show most significant changes visualization
        if compared_features:
            st.markdown("### 📊 Most Significant Changes")
            
            # Use the chart utility to create the bar chart
            if USE_CUSTOM_MODULES:
                fig = create_change_bar_chart(compared_features, feature_descriptions)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Use the fallback table display instead of interactive chart
                st.write("Top changes between visits:")
                change_data = create_change_bar_chart(compared_features, feature_descriptions)
                st.dataframe(change_data)
        
        # Create categories for better organization
        feature_categories = {
            "Cognitive Tests": ["CDRSB", "MMSE", "MOCA", "ADAS11", "ADAS13", "FAQ"],
            "Memory Tests": ["RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting", "LDELTOTAL"],
            "Brain Measurements": ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"],
            "Biomarkers": ["ABETA", "TAU", "PTAU", "APOE4"],
            "Demographics & Other": ["AGE", "PTGENDER", "PTEDUCAT"]
        }
        
        # Display detailed comparison tables by category
        st.markdown("### 📋 Detailed Comparison by Category")
        
        # Create tabs for different categories
        tabs = st.tabs(list(feature_categories.keys()))
        
        for i, (category, features) in enumerate(feature_categories.items()):
            with tabs[i]:
                # Filter features for this category
                category_features = [f for f in features if f in visit1_features or f in visit2_features]
                
                if not category_features:
                    st.info(f"No {category.lower()} data available")
                    continue
                
                # Create DataFrame for comparison
                rows = []
                for feature in category_features:
                    val1 = visit1_features.get(feature, None)
                    val2 = visit2_features.get(feature, None)
                    
                    if val1 is not None and val2 is not None:
                        val1_float = float(val1)
                        val2_float = float(val2)
                        abs_change = val2_float - val1_float
                        pct_change = ((val2_float - val1_float) / val1_float) * 100 if val1_float != 0 else 0
                        
                        # Determine if this change is improvement or decline
                        is_improvement = (feature in decline_if_higher and abs_change < 0) or (feature not in decline_if_higher and abs_change > 0)
                        
                        # Create status indicator
                        if abs(pct_change) < 1:  # Less than 1% change
                            status = "No Change"
                            status_class = "highlight-same"
                            arrow = "→"
                        elif is_improvement:
                            status = "Improved"
                            status_class = "highlight-improvement"
                            arrow = "↑"
                        else:
                            status = "Declined"
                            status_class = "highlight-decline"
                            arrow = "↓"
                        
                        # Get description for tooltip
                        description = feature_descriptions.get(feature, "")
                        
                        rows.append({
                            "Feature": feature,
                            "Description": description,
                            "First Visit": f"{val1_float:.2f}",
                            "Second Visit": f"{val2_float:.2f}",
                            "Absolute Change": abs_change,
                            "Percent Change": pct_change,
                            "Status": status,
                            "Status Class": status_class,
                            "Arrow": arrow,
                            "Is Improvement": is_improvement
                        })
                
                if not rows:
                    st.info(f"No comparable {category.lower()} data available")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(rows)
                
                # Add a statistical summary of changes for this category
                with st.expander(f"📊 Statistical Summary for {category}"):
                    # Calculate metrics with proper numeric conversion
                    total_changes = len(df)
                    
                    # Ensure Absolute Change is numeric
                    df['Absolute Change'] = pd.to_numeric(df['Absolute Change'], errors='coerce').fillna(0)
                    
                    positive_changes = sum(df["Absolute Change"] > 0)
                    negative_changes = sum(df["Absolute Change"] < 0)
                    no_changes = sum(df["Absolute Change"] == 0)
                    avg_change = df["Absolute Change"].mean()
                    max_positive = df["Absolute Change"].max()
                    max_negative = df["Absolute Change"].min()
                    
                    # Display summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Total measures:** {total_changes}")
                        st.markdown(f"**Positive changes:** {positive_changes}")
                        st.markdown(f"**Negative changes:** {negative_changes}")
                        st.markdown(f"**No change:** {no_changes}")
                        
                    with col2:
                        st.markdown(f"**Average change:** {avg_change:.2f}")
                        st.markdown(f"**Maximum positive change:** {max_positive:.2f}")
                        st.markdown(f"**Maximum negative change:** {max_negative:.2f}")
                    
                    # Add interpretation guidance based on the category
                    if category == "Cognitive Tests":
                        st.markdown("**Interpretation guide:** Lower scores on CDRSB, ADAS11, ADAS13, and FAQ indicate better cognitive function, while higher scores on MMSE and MOCA indicate better function.")
                    elif category == "Brain Measurements":
                        st.markdown("**Interpretation guide:** Lower values for Ventricles indicate better brain health, while higher values for Hippocampus, Entorhinal, Fusiform, MidTemp, and WholeBrain typically indicate better brain preservation.")
                    elif category == "Memory Tests":
                        st.markdown("**Interpretation guide:** Higher scores on RAVLT_immediate, RAVLT_learning, and LDELTOTAL indicate better memory function, while lower scores on RAVLT_forgetting and RAVLT_perc_forgetting indicate less memory loss.")
                    elif category == "Biomarkers":
                        st.markdown("**Interpretation guide:** Higher values for ABETA and lower values for TAU and PTAU typically indicate better brain health. APOE4 is a genetic risk factor (0, 1, or 2 alleles).")
                
                # Replace the HTML table with Streamlit's native dataframe rendering
                st.markdown(f"#### {category} Comparison")
                
                # Create a more readable display dataframe with descriptions instead of status
                display_df = df[["Feature", "Description", "First Visit", "Second Visit", "Absolute Change"]]
                
                # Limit description length to ensure readability
                display_df["Description"] = display_df["Description"].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
                
                # Apply enhanced styling based on change magnitude
                def style_dataframe(df):
                    # Create a deep copy to avoid SettingWithCopyWarning
                    formatted_df = df.copy(deep=True)
                    
                    # Create a new column for formatted absolute change instead of modifying the original
                    if 'Absolute Change' in formatted_df.columns:
                        # First convert to numeric to ensure proper type
                        formatted_df['Absolute Change'] = pd.to_numeric(formatted_df['Absolute Change'], errors='coerce')
                        
                        # Create a new column for the formatted values
                        formatted_df['Change Display'] = formatted_df['Absolute Change'].apply(
                            lambda value: f"↑ +{value:.2f}" if value > 0 else f"↓ {value:.2f}" if value < 0 else f"→ {value:.2f}"
                        )
                        
                        # Keep original Absolute Change for sorting/calculations
                        # But use the formatted column for display
                    
                    return formatted_df
                
                # Format the data first
                display_df = style_dataframe(display_df)
                
                # Reorder columns if needed - do this BEFORE applying styles
                if 'Change Display' in display_df.columns:
                    # Create a new column order with Change Display in place of Absolute Change
                    display_columns = [col for col in display_df.columns if col != 'Absolute Change']
                    
                    # Make sure Change Display is in the right position (after Second Visit)
                    if 'First Visit' in display_columns and 'Second Visit' in display_columns:
                        # Get the indices
                        col_list = display_columns.copy()
                        second_visit_idx = col_list.index('Second Visit')
                        
                        # If Change Display is in the wrong position, move it
                        if 'Change Display' in col_list:
                            col_list.remove('Change Display')
                            col_list.insert(second_visit_idx + 1, 'Change Display')
                        
                        # Reorder the DataFrame directly
                        display_df = display_df[col_list]
                
                # Define styling functions for each column
                def style_feature(val):
                    return 'font-weight: bold'
                
                def style_description(val):
                    return 'color: #666; font-style: italic'
                
                def style_change_display(val):
                    # Style based on the arrow in the formatted string
                    try:
                        if '↑' in str(val):  # Positive change
                            return 'color: green; background-color: rgba(209, 250, 229, 0.4)'
                        elif '↓' in str(val):  # Negative change
                            return 'color: red; background-color: rgba(254, 226, 226, 0.4)'
                        else:  # No change
                            return 'color: gray; background-color: rgba(229, 231, 235, 0.3)'
                    except:
                        return ''
                
                # Apply styles to the properly ordered DataFrame
                styled_df = display_df.style
                
                # Apply column-specific styling 
                if 'Feature' in display_df.columns:
                    styled_df = styled_df.map(style_feature, subset=['Feature'])
                
                if 'Description' in display_df.columns:
                    styled_df = styled_df.map(style_description, subset=['Description'])
                
                if 'Change Display' in display_df.columns:
                    styled_df = styled_df.map(style_change_display, subset=['Change Display'])
                elif 'Absolute Change' in display_df.columns:
                    # If we don't have the formatted column for some reason, use original
                    styled_df = styled_df.map(lambda v: style_change_display(str(v)), subset=['Absolute Change'])
                
                # Display the styled dataframe
                st.dataframe(styled_df, use_container_width=True)
                
                # Show visualization for this category if we have enough data
                if len(rows) > 1:
                    st.markdown("#### Visualization")
                    
                    # Create better visualizations that highlight changes more effectively
                    try:
                        import plotly.graph_objects as go
                        import plotly.express as px
                        
                        # Process data for visualization - ensure proper numeric conversion
                        feature_names = df["Feature"].tolist()
                        
                        # Convert string values to float for visualization
                        visit1_values = df["First Visit"].apply(lambda x: float(str(x).replace(',', ''))).tolist()
                        visit2_values = df["Second Visit"].apply(lambda x: float(str(x).replace(',', ''))).tolist()
                        
                        # Ensure Absolute Change is numeric
                        abs_changes = pd.to_numeric(df["Absolute Change"], errors='coerce').fillna(0).tolist()
                        
                        # Create a tab view for different chart types
                        chart_tabs = st.tabs(["Change Magnitude", "Before vs After", "Radar View"])
                        
                        # Tab 1: Bar chart showing magnitude of changes
                        with chart_tabs[0]:
                            # Create a bar chart showing the magnitude of changes
                            fig = px.bar(
                                x=feature_names, 
                                y=abs_changes,
                                title=f"Change Magnitude in {category}",
                                labels={"x": "Feature", "y": "Change"},
                                color=[
                                    "Positive" if change > 0 else "Negative" if change < 0 else "No Change"
                                    for change in abs_changes
                                ],
                                color_discrete_map={
                                    "Positive": "#4BB543", 
                                    "Negative": "#FF3333",
                                    "No Change": "#808080"
                                }
                            )
                            fig.update_layout(
                                height=400, 
                                margin={"t": 50, "b": 50, "l": 50, "r": 50},
                                legend_title_text="Change Direction"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation for the chart
                            st.caption("This chart shows the magnitude of change for each feature. Positive changes are shown in green, negative changes in red.")
                            
                            # Add interpretation note about the measures
                            if any(f in decline_if_higher for f in feature_names):
                                st.info(f"Note: For this category, the effect of changes depends on the specific measure. Please refer to the interpretation guide for details on which direction is better for each measure.")
                        
                        # Tab 2: Before vs After bar chart
                        with chart_tabs[1]:
                            # Create a grouped bar chart for before vs after
                            data = pd.DataFrame({
                                "Feature": feature_names * 2,
                                "Value": visit1_values + visit2_values,
                                "Visit": ["First Visit"] * len(feature_names) + ["Second Visit"] * len(feature_names)
                            })
                            
                            fig = px.bar(
                                data,
                                x="Feature", 
                                y="Value",
                                color="Visit",
                                barmode="group",
                                title=f"Before vs After Comparison for {category}",
                                color_discrete_map={"First Visit": "#4e73df", "Second Visit": "#f6c23e"}
                            )
                            fig.update_layout(
                                height=400, 
                                margin={"t": 50, "b": 50, "l": 50, "r": 50},
                                legend_title_text="Visit"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation about interpretation
                            if any(f in decline_if_higher for f in feature_names):
                                st.info("Note: For some measures (like CDRSB, ADAS11, Ventricles), lower values are better.")
                        
                        # Tab 3: Radar chart for comprehensive view
                        with chart_tabs[2]:
                            if len(feature_names) >= 3:  # Radar charts need at least 3 points
                                # Create radar chart
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=visit1_values,
                                    theta=feature_names,
                                    fill='toself',
                                    name='First Visit'
                                ))
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=visit2_values,
                                    theta=feature_names,
                                    fill='toself',
                                    name='Second Visit'
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                        )
                                    ),
                                    title=f"Radar View of {category} Changes",
                                    height=500,
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanation about interpretation
                                if any(f in decline_if_higher for f in feature_names):
                                    st.info("Note: In this radar chart, points further from center indicate higher values, which may be better or worse depending on the measure.")
                            else:
                                st.info("Radar chart requires at least 3 features for meaningful visualization.")
                    
                    except Exception as e:
                        # Fallback to simple visualization if advanced charts fail
                        st.error(f"Error generating advanced visualizations: {e}")
                        
                        # Use chart utility functions for basic visualizations if available
                        if len(rows) >= 3 and category != "Brain Measurements":
                            if USE_CUSTOM_MODULES:
                                radar_fig = create_radar_chart(df, category, decline_if_higher)
                                if radar_fig:
                                    st.plotly_chart(radar_fig, use_container_width=True)
                            else:
                                # Fallback for when chart module isn't available
                                st.write("Feature comparison:")
                                st.dataframe(df[["Feature", "First Visit", "Second Visit", "Status"]])
                        
                        # For brain measurements, show a side by side bar chart
                        if category == "Brain Measurements" and len(rows) > 0:
                            if USE_CUSTOM_MODULES:
                                brain_fig = create_brain_measurement_chart(df)
                                st.plotly_chart(brain_fig, use_container_width=True)
                            else:
                                # Fallback for when chart module isn't available
                                st.write("Brain measurement comparison:")
                                st.dataframe(df[["Feature", "First Visit", "Second Visit", "Absolute Change"]])
        
        # Add download option for comparison data
        comparison_csv = pd.DataFrame({
            'Feature': list(set(visit1_features.keys()) | set(visit2_features.keys())),
            f'Visit 1 ({visit1_date})': [visit1_features.get(f, "N/A") for f in set(visit1_features.keys()) | set(visit2_features.keys())],
            f'Visit 2 ({visit2_date})': [visit2_features.get(f, "N/A") for f in set(visit1_features.keys()) | set(visit2_features.keys())]
        }).to_csv(index=False)
        
        st.download_button(
            label="📥 Download Comparison Data",
            data=comparison_csv,
            file_name=f"visit_comparison_{visit1_date}_to_{visit2_date}.csv",
            mime="text/csv"
        )
        
        # Add AI-generated insights
        st.markdown("### 🤖 AI-Generated Clinical Assessment")
        with st.spinner("Generating analysis of changes between visits..."):
            try:
                # Prepare prompt for the AI with more clinical details
                prompt = f"""
                You are a clinical expert in Alzheimer's disease progression. Analyze the changes between two patient visits and provide clinical insights.
                
                Patient Information:
                - Name: {patient_info['full_name']}
                - Age: {datetime.now().year - patient_info['birth_date'].year} years
                - Gender: {patient_info['gender']}
                
                First Visit ({visit1_date}):
                - Diagnosis: {visit1_pred}
                - Confidence: {visit1_conf:.1f}%
                
                Second Visit ({visit2_date}):
                - Diagnosis: {visit2_pred}
                - Confidence: {visit2_conf:.1f}%
                
                Time Between Visits: {days_between} days
                
                Key Changes in Clinical Features:
                """
                
                # List all significant changes for AI context
                important_features = ["CDRSB", "MMSE", "ADAS11", "ADAS13", "RAVLT_immediate", 
                                     "Hippocampus", "Entorhinal", "Ventricles", "Fusiform",
                                     "ABETA", "TAU", "PTAU", "APOE4"]
                
                for feature in important_features:
                    if feature in visit1_features and feature in visit2_features:
                        val1 = float(visit1_features[feature])
                        val2 = float(visit2_features[feature])
                        pct_change = ((val2 - val1) / val1) * 100 if val1 != 0 else 0
                        
                        # Add feature description
                        desc = feature_descriptions.get(feature, "")
                        short_desc = desc.split("-")[0].strip() if "-" in desc else desc
                        
                        prompt += f"- {feature}: {val1:.2f} → {val2:.2f} ({pct_change:+.1f}%) - {short_desc}\n"
                
                prompt += """
                Please provide:
                1. A clinical assessment of the changes observed between visits
                2. Interpretation of the most significant biomarker or cognitive test changes
                3. Recommendations for clinical management based on these changes
                4. Prognosis based on the observed trajectory
                
                Format your response in clear sections with concise, clinically-relevant insights.
                """
                
                response = model.generate_content(prompt)
                analysis = response.text
                
                # Display the AI analysis in a styled container
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
                           border-left: 5px solid #4e73df; margin-top: 20px;">
                """, unsafe_allow_html=True)
                
                st.markdown(analysis)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Option to save analysis to patient records
                if st.button("💾 Save Analysis to Medical Records"):
                    from doctor_view import add_medical_record
                    diagnosis = f"Visit Comparison Analysis: {visit1_date} to {visit2_date}"
                    
                    summary = f"""
                    VISIT COMPARISON ANALYSIS
                    
                    First Visit: {visit1_date} - Diagnosis: {visit1_pred} ({visit1_conf:.1f}%)
                    Second Visit: {visit2_date} - Diagnosis: {visit2_pred} ({visit2_conf:.1f}%)
                    Time Between Visits: {days_between} days
                    
                    Key Changes:
                    - Improvements: {len(improvements)}
                    - Declines: {len(declines)}
                    
                    AI Assessment:
                    {analysis}
                    """
                    
                    if add_medical_record(patient_info['patient_id'], diagnosis, summary):
                        st.success("✅ Analysis saved to medical records")
                    else:
                        st.error("❌ Failed to save analysis")
                
            except Exception as e:
                st.error(f"Error generating AI insights: {e}")
    
    except Exception as e:
        st.error(f"Error processing visit comparison data: {e}")
        import traceback
        st.error(traceback.format_exc())
    
    return 