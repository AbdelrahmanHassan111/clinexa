"""
Chart utility functions for the visit comparison feature.
These functions help create the visualizations for comparing patient visits.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def create_change_bar_chart(compared_features, feature_descriptions, limit=8):
    """
    Create a horizontal bar chart showing the most significant changes
    
    Args:
        compared_features: List of dictionaries with feature comparison data
        feature_descriptions: Dictionary mapping feature names to descriptions
        limit: Maximum number of features to show
        
    Returns:
        Plotly figure object
    """
    # Sort features by absolute percent change
    top_changes = sorted(compared_features, key=lambda x: abs(x['pct_change']), reverse=True)[:limit]
    
    # Prepare data for chart
    chart_data = pd.DataFrame(top_changes)
    chart_data['color'] = chart_data['is_decline'].apply(lambda x: '#dc3545' if x else '#28a745')
    chart_data['feature_name'] = chart_data['feature'].apply(
        lambda x: f"{x}: {feature_descriptions.get(x, '').split('-')[0] if feature_descriptions.get(x, '') else ''}"
    )
    
    # Create a horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=chart_data['feature_name'],
        x=chart_data['pct_change'],
        orientation='h',
        marker_color=chart_data['color'],
        text=chart_data['pct_change'].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside',
    ))
    
    fig.update_layout(
        title='Percent Change in Key Measurements',
        xaxis_title='Percent Change (%)',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    
    return fig

def create_radar_chart(df, feature_category, decline_if_higher):
    """
    Create a radar chart comparing feature values between visits
    
    Args:
        df: DataFrame with feature comparison data
        feature_category: Category name for the chart title
        decline_if_higher: List of features where higher values indicate decline
        
    Returns:
        Plotly figure object
    """
    # Create normalized data for radar chart
    radar_data = {}
    
    for _, row in df.iterrows():
        feature = row["Feature"]
        
        # Skip features that shouldn't be in radar charts
        if feature in ["AGE", "PTGENDER", "PTEDUCAT"]:
            continue
            
        visit1_val = float(row["First Visit"])
        visit2_val = float(row["Second Visit"])
        
        radar_data[feature] = {
            "Visit 1": visit1_val,
            "Visit 2": visit2_val,
            "Is Higher Better": feature not in decline_if_higher
        }
    
    if not radar_data:
        return None
    
    # Create radar chart using Plotly
    categories = list(radar_data.keys())
    
    # Prepare data in the format needed for radar chart
    r1 = [radar_data[cat]["Visit 1"] for cat in categories]
    r2 = [radar_data[cat]["Visit 2"] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=r1,
        theta=categories,
        fill='toself',
        name='First Visit',
        line_color='#007bff',
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=r2,
        theta=categories,
        fill='toself',
        name='Second Visit',
        line_color='#dc3545',
        opacity=0.7
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            ),
        ),
        title=f"{feature_category} Comparison",
        showlegend=True
    )
    
    return fig

def create_brain_measurement_chart(df):
    """
    Create a side-by-side bar chart for brain measurements
    
    Args:
        df: DataFrame with brain measurement comparison data
        
    Returns:
        Plotly figure object
    """
    brain_data = df.copy()
    brain_data["First Visit"] = brain_data["First Visit"].astype(float)
    brain_data["Second Visit"] = brain_data["Second Visit"].astype(float)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='First Visit',
        x=brain_data["Feature"],
        y=brain_data["First Visit"],
        marker_color='#4e73df'
    ))
    fig.add_trace(go.Bar(
        name='Second Visit',
        x=brain_data["Feature"],
        y=brain_data["Second Visit"],
        marker_color='#1cc88a'
    ))
    
    fig.update_layout(
        title='Brain Measurements Comparison',
        barmode='group',
        xaxis_title='Brain Region',
        yaxis_title='Volume (mm³)'
    )
    
    return fig

def create_timeline_chart(visit_history, current_visits):
    """
    Create a timeline chart showing patient's visit history and highlighting current comparison
    
    Args:
        visit_history: List of dictionaries with visit data
        current_visits: List of two visit IDs being compared
        
    Returns:
        Plotly figure object
    """
    if not visit_history:
        return None
    
    # Prepare data for timeline
    df = pd.DataFrame(visit_history)
    
    # Add color coding for the current comparison visits
    df['color'] = 'gray'
    df.loc[df['analysis_id'].isin(current_visits), 'color'] = 'blue'
    
    # Create the timeline
    fig = px.scatter(
        df, 
        x="analyzed_at", 
        y=[1] * len(df),  # All on same y level
        color="color",
        color_discrete_map={"blue": "#1e88e5", "gray": "#b0bec5"},
        hover_name="prediction",
        hover_data=["confidence_score", "analyzed_at"],
        title="Patient Visit Timeline",
        labels={"analyzed_at": "Date", "y": ""}
    )
    
    # Customize the timeline appearance
    fig.update_traces(marker=dict(size=15, symbol='diamond'))
    fig.update_layout(
        showlegend=False,
        yaxis=dict(visible=False),
        height=150,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Add connecting lines
    for i in range(len(df) - 1):
        fig.add_shape(
            type="line",
            x0=df.iloc[i]["analyzed_at"],
            y0=1,
            x1=df.iloc[i+1]["analyzed_at"],
            y1=1,
            line=dict(color="#b0bec5", width=2, dash="dot")
        )
    
    # Highlight the comparison visits
    if len(current_visits) == 2:
        # Find the two visits
        visit1 = df[df['analysis_id'] == current_visits[0]].iloc[0]
        visit2 = df[df['analysis_id'] == current_visits[1]].iloc[0]
        
        # Add a highlighted connection line between them
        fig.add_shape(
            type="line",
            x0=visit1["analyzed_at"],
            y0=1,
            x1=visit2["analyzed_at"],
            y1=1,
            line=dict(color="#1e88e5", width=4)
        )
    
    return fig 