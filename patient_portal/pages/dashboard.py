import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from ..utils.db import (
    get_patient_info, 
    get_patient_appointments, 
    get_patient_records, 
    get_patient_cognitive_scores,
    get_mri_scans
)

def patient_dashboard(patient_id):
    """Main dashboard for the patient portal."""
    
    # Get patient info
    patient_info = get_patient_info(patient_id)
    if not patient_info:
        st.error("Could not retrieve patient information")
        return
    
    # Dashboard header
    st.markdown(f"""
    <div class="dashboard-header">
        <h1>Welcome, {patient_info.get('first_name', 'Patient')}</h1>
        <p>Your personalized health dashboard with key metrics and upcoming appointments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # First row - Summary cards
    col1, col2, col3 = st.columns(3)
    
    # Get data
    appointments = get_patient_appointments(patient_id, limit=5)
    records = get_patient_records(patient_id, limit=10)
    scores = get_patient_cognitive_scores(patient_id, limit=10)
    mri_scans = get_mri_scans(patient_id, limit=5)
    
    # Today's date for comparison
    today = datetime.now().date()
    
    # Upcoming appointments
    with col1:
        st.markdown('<div class="section-title">Upcoming Appointments</div>', unsafe_allow_html=True)
        
        upcoming_count = 0
        for appt in appointments:
            appt_date = datetime.strptime(appt['formatted_date'], "%Y-%m-%d").date()
            if appt_date >= today and appt['status'] == 'scheduled':
                upcoming_count += 1
                
                # Format time
                time_str = appt['formatted_time']
                
                # Status badge based on proximity
                days_until = (appt_date - today).days
                status_class = "scheduled"
                if days_until <= 1:
                    status_class = "danger"  # Red for tomorrow/today
                elif days_until <= 3:
                    status_class = "warning"  # Orange for soon
                
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <div class="card-title">{appt_date.strftime('%a, %b %d')}, {time_str}</div>
                            <div style="font-weight: 500; margin-bottom: 0.5rem;">{appt.get('appointment_type', 'Consultation')}</div>
                            <div class="card-content">Dr. {appt.get('doctor_name', 'Unknown')}</div>
                        </div>
                        <span class="status-tag {status_class}">{days_until} day{'s' if days_until != 1 else ''}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if upcoming_count == 0:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📅</div>
                <div class="card-title">No Upcoming Appointments</div>
                <div class="card-content">Schedule your next visit</div>
                <br>
                <div>
                    <a href="#" onclick="document.querySelector('[data-testid=stSidebar]').querySelector('button:contains(Appointments)').click(); return false;" style="color: #3b82f6; text-decoration: none; font-weight: 500;">
                        Schedule Now
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent health updates
    with col2:
        st.markdown('<div class="section-title">Health Updates</div>', unsafe_allow_html=True)
        
        if records:
            # Display most recent record
            recent_record = records[0]
            visit_date = datetime.strptime(recent_record['formatted_date'], "%Y-%m-%d").date()
            days_ago = (today - visit_date).days
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Recent Medical Record</div>
                <div class="record-date">{visit_date.strftime('%B %d, %Y')} ({days_ago} days ago)</div>
                <div style="font-weight: 500; margin-bottom: 0.5rem;">{recent_record.get('diagnosis', 'Check-up')}</div>
                <div class="card-content">{recent_record.get('clinical_notes', '')[:100]}...</div>
                <div style="margin-top: 1rem; text-align: right;">
                    <a href="#" onclick="document.querySelector('[data-testid=stSidebar]').querySelector('button:contains(Medical Records)').click(); return false;" style="color: #3b82f6; text-decoration: none; font-weight: 500;">
                        View Full Record
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # MRI scan if available
            if mri_scans:
                recent_mri = mri_scans[0]
                scan_date = datetime.strptime(recent_mri['formatted_date'], "%Y-%m-%d").date()
                
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Recent Brain Scan</div>
                    <div class="record-date">{scan_date.strftime('%B %d, %Y')}</div>
                    <div style="font-weight: 500; margin-bottom: 0.5rem;">{recent_mri.get('scan_type', 'MRI Scan')}</div>
                    <div class="card-content">{recent_mri.get('scan_notes', 'No notes available')[:100]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📋</div>
                <div class="card-title">No Recent Updates</div>
                <div class="card-content">Your medical history will appear here</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Cognitive health
    with col3:
        st.markdown('<div class="section-title">Cognitive Health</div>', unsafe_allow_html=True)
        
        if scores and len(scores) > 1:
            # Get the latest score
            latest_score = scores[0]
            
            # Create a mini trend visualization
            dates = [datetime.strptime(score['formatted_date'], "%Y-%m-%d").date() for score in scores[:5]]
            if 'confidence_score' in latest_score:
                values = [float(score['confidence_score']) * 100 for score in scores[:5]]
                metric_title = "Cognitive Health Score"
                metric_value = f"{values[0]:.1f}%"
                
                # Determine trend
                trend = values[0] - values[-1]
                
                # Set status class based on score
                status_class = "good"
                if values[0] < 70:
                    status_class = "danger"
                elif values[0] < 85:
                    status_class = "warning"
                
                st.markdown(f"""
                <div class="health-metric {status_class}">
                    <div class="health-metric-title">{metric_title}</div>
                    <div class="health-metric-value">{metric_value}</div>
                    <div class="health-metric-description">
                        {trend:+.1f}% compared to previous assessment
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create small trend chart
                fig = px.line(
                    x=dates, y=values, 
                    markers=True, line_shape='spline',
                    template='plotly_white',
                    color_discrete_sequence=['#3b82f6']
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=200,
                    showlegend=False,
                    xaxis_title="", yaxis_title="",
                    xaxis_showgrid=False, yaxis_showgrid=True,
                    yaxis_range=[min(values)-5, max(values)+5]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Create a generic health metric if no confidence score available
                prediction = latest_score.get('prediction', 'No data')
                
                # Determine status class based on prediction
                status_class = "good"
                if "Demented" in prediction:
                    status_class = "danger"
                elif "Converted" in prediction or "Mild" in prediction:
                    status_class = "warning"
                
                st.markdown(f"""
                <div class="health-metric {status_class}">
                    <div class="health-metric-title">Cognitive Assessment</div>
                    <div class="health-metric-value">{prediction}</div>
                    <div class="health-metric-description">
                        Last updated: {dates[0].strftime('%b %d, %Y')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🧠</div>
                <div class="card-title">No Cognitive Data</div>
                <div class="card-content">Your cognitive assessments will appear here after your first evaluation</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row - Interactive charts
    st.markdown('<div class="section-title">Health Metrics</div>', unsafe_allow_html=True)
    
    if scores and len(scores) > 2:
        # Create sample data if actual metrics are missing
        try:
            # Extract cognitive features from scores
            feature_names = []
            feature_values = []
            
            if 'MMSE' in scores[0]:
                feature_names.append('MMSE')
                feature_values.append([float(score.get('MMSE', 0)) for score in scores if 'MMSE' in score])
            
            if 'CDRSB' in scores[0]:
                feature_names.append('CDRSB')
                feature_values.append([float(score.get('CDRSB', 0)) for score in scores if 'CDRSB' in score])
            
            if 'Hippocampus' in scores[0]:
                feature_names.append('Hippocampus')
                feature_values.append([float(score.get('Hippocampus', 0)) for score in scores if 'Hippocampus' in score])
            
            # If we have some real feature data
            if feature_names:
                # Create a comprehensive chart
                fig = go.Figure()
                
                for i, name in enumerate(feature_names):
                    if len(feature_values[i]) > 1:
                        # Normalize values for comparison
                        values = feature_values[i]
                        normalized = [(v - min(values)) / (max(values) - min(values) + 0.0001) * 100 for v in values]
                        
                        fig.add_trace(go.Scatter(
                            x=dates[:len(values)],
                            y=normalized,
                            mode='lines+markers',
                            name=name,
                            line=dict(width=3),
                            marker=dict(size=8)
                        ))
                
                fig.update_layout(
                    title="Cognitive Health Trends (Normalized)",
                    xaxis_title="Date",
                    yaxis_title="Score (Normalized)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_white",
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create a sample dashboard chart with simulated data
                create_sample_chart()
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            create_sample_chart()
    else:
        # Create a sample dashboard chart with simulated data
        create_sample_chart()
    
    # Third row - Recommendations
    st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧘</div>
            <div class="card-title">Mental Exercise</div>
            <div class="card-content">
                Regular cognitive stimulation can help maintain brain health. Try puzzles, reading, or learning a new skill for 30 minutes daily.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🍎</div>
            <div class="card-title">Diet & Nutrition</div>
            <div class="card-content">
                Maintain a Mediterranean-style diet rich in vegetables, fruits, whole grains, and omega-3 fatty acids to support brain health.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚶</div>
            <div class="card-title">Physical Activity</div>
            <div class="card-content">
                Aim for at least 150 minutes of moderate aerobic activity each week to improve blood flow to the brain and reduce risk factors.
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_sample_chart():
    """Create a sample chart with simulated data for the dashboard."""
    # Generate sample dates for the past 6 months
    today = datetime.now().date()
    dates = [(today - timedelta(days=30*i)) for i in range(6)]
    
    # Generate sample cognitive scores (improving trend)
    cognitive_scores = [70 + i*2 + np.random.randint(-3, 4) for i in range(6)]
    cognitive_scores.reverse()  # Make it increasing (improving) over time
    
    # Generate sample memory scores (improving trend)
    memory_scores = [65 + i*3 + np.random.randint(-5, 6) for i in range(6)]
    memory_scores.reverse()
    
    # Generate sample attention scores (stable with slight improvement)
    attention_scores = [75 + i*1 + np.random.randint(-4, 5) for i in range(6)]
    attention_scores.reverse()
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cognitive_scores,
        mode='lines+markers',
        name='Overall Cognitive',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=memory_scores,
        mode='lines+markers',
        name='Memory',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=attention_scores,
        mode='lines+markers',
        name='Attention',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Simulated Health Trends (Sample Data)",
        xaxis_title="Date",
        yaxis_title="Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=400
    )
    
    # Add annotation explaining this is sample data
    fig.add_annotation(
        text="*This is simulated data for demonstration purposes only",
        xref="paper", yref="paper",
        x=0, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    st.plotly_chart(fig, use_container_width=True) 