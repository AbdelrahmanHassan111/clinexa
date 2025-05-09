import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from ..utils.db import get_patient_records, get_patient_cognitive_scores, get_mri_scans

def medical_records_page(patient_id):
    """Display patient medical records with enhanced UI."""
    
    # Page header
    st.markdown("""
    <div class="dashboard-header">
        <h1>Medical Records</h1>
        <p>Review your medical history, diagnoses, and test results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different types of medical records
    tab1, tab2, tab3 = st.tabs(["Clinical Records", "Brain Scans", "Cognitive Assessments"])
    
    # Clinical Records Tab
    with tab1:
        clinical_records_tab(patient_id)
    
    # Brain Scans Tab
    with tab2:
        brain_scans_tab(patient_id)
    
    # Cognitive Assessments Tab
    with tab3:
        cognitive_assessments_tab(patient_id)

def clinical_records_tab(patient_id):
    """Display clinical records in a modern UI."""
    
    # Get patient records
    records = get_patient_records(patient_id)
    
    if not records:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
            <h3>No Medical Records Found</h3>
            <p>Your clinical history will appear here after your first appointment.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Add search and filter options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search records", placeholder="Search by diagnosis, doctor, or notes")
    
    with col2:
        sort_option = st.selectbox(
            "Sort by",
            ["Newest first", "Oldest first", "Doctor name"],
            index=0
        )
    
    # Process records based on search and sort
    filtered_records = process_records(records, search_term, sort_option)
    
    # Display timeline of medical records
    if filtered_records:
        # Show pagination controls if there are many records
        records_per_page = 5
        total_pages = (len(filtered_records) + records_per_page - 1) // records_per_page
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                page = st.slider("Page", 1, total_pages, 1)
            
            start_idx = (page - 1) * records_per_page
            end_idx = min(start_idx + records_per_page, len(filtered_records))
            page_records = filtered_records[start_idx:end_idx]
        else:
            page_records = filtered_records
        
        st.markdown("<div class='section-title'>Clinical History</div>", unsafe_allow_html=True)
        for i, record in enumerate(page_records):
            # Format the record in a card with expandable details
            display_clinical_record(record, i)
    else:
        st.info("No records match your search criteria.")

def process_records(records, search_term, sort_option):
    """Process and filter records based on search terms and sort options."""
    # Filter by search term if provided
    if search_term:
        search_term = search_term.lower()
        filtered = []
        for record in records:
            if (search_term in record.get('diagnosis', '').lower() or
                search_term in record.get('clinical_notes', '').lower() or
                search_term in record.get('doctor_name', '').lower()):
                filtered.append(record)
        records = filtered
    
    # Sort records based on selected option
    if sort_option == "Newest first":
        records = sorted(records, key=lambda x: x.get('visit_date', ''), reverse=True)
    elif sort_option == "Oldest first":
        records = sorted(records, key=lambda x: x.get('visit_date', ''))
    elif sort_option == "Doctor name":
        records = sorted(records, key=lambda x: x.get('doctor_name', ''))
    
    return records

def display_clinical_record(record, index):
    """Display a single clinical record in a modern card format."""
    # Format date
    try:
        visit_date = datetime.strptime(record['formatted_date'], "%Y-%m-%d").date()
        date_display = visit_date.strftime("%B %d, %Y")
    except:
        date_display = record.get('formatted_date', 'Unknown date')
    
    doctor_name = record.get('doctor_name', 'Unknown doctor')
    diagnosis = record.get('diagnosis', 'No diagnosis recorded')
    clinical_notes = record.get('clinical_notes', 'No clinical notes available')
    
    # Determine if there's a recommendation
    has_recommendation = "recommend" in clinical_notes.lower() or "follow up" in clinical_notes.lower()
    
    # Create the card
    st.markdown(f"""
    <div class="record-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div class="record-date">{date_display}</div>
            <div style="font-size: 0.875rem; color: #6b7280;">Dr. {doctor_name}</div>
        </div>
        <div class="record-title">{diagnosis}</div>
        <div class="record-content">
            {clinical_notes[:200]}{'...' if len(clinical_notes) > 200 else ''}
        </div>
    """, unsafe_allow_html=True)
    
    # Add expandable section for full details
    if st.button(f"View Full Record #{index}", key=f"view_record_{index}"):
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 1rem; background-color: #f3f4f6; border-radius: 0.5rem;">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Clinical Notes</div>
            <div style="white-space: pre-line;">{clinical_notes}</div>
            
            {'<div style="margin-top: 1rem; padding: 0.75rem; background-color: #ecfdf5; border-radius: 0.5rem; border-left: 4px solid #10b981;"><div style="font-weight: 600; margin-bottom: 0.5rem;">Recommendations</div>' + extract_recommendations(clinical_notes) + '</div>' if has_recommendation else ''}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def extract_recommendations(notes):
    """Extract recommendations from clinical notes."""
    lines = notes.split(".")
    recommendations = []
    
    for line in lines:
        lower_line = line.lower()
        if ("recommend" in lower_line or 
            "advised" in lower_line or 
            "follow up" in lower_line or
            "suggested" in lower_line):
            recommendations.append(line.strip() + ".")
    
    if recommendations:
        return "<br>".join(recommendations)
    else:
        return "No specific recommendations found."

def brain_scans_tab(patient_id):
    """Display brain MRI scans with visualization."""
    
    # Get MRI scans
    scans = get_mri_scans(patient_id)
    
    if not scans:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <h3>No Brain Scans Found</h3>
            <p>Your MRI scans will appear here after they are uploaded by your doctor.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Convert to dataframe for better handling
    scans_df = pd.DataFrame(scans)
    
    # Display statistics
    st.markdown("<div class='section-title'>Brain Scan Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="health-metric">
            <div class="health-metric-title">Total Scans</div>
            <div class="health-metric-value">{len(scans)}</div>
            <div class="health-metric-description">
                Last scan: {scans_df['formatted_date'].iloc[0] if not scans_df.empty else 'None'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'is_processed' in scans_df.columns:
            processed_count = scans_df['is_processed'].sum()
            st.markdown(f"""
            <div class="health-metric">
                <div class="health-metric-title">Analyzed Scans</div>
                <div class="health-metric-value">{processed_count}</div>
                <div class="health-metric-description">
                    {processed_count / len(scans) * 100:.0f}% of scans have been analyzed
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="health-metric">
                <div class="health-metric-title">Analyzed Scans</div>
                <div class="health-metric-value">N/A</div>
                <div class="health-metric-description">
                    Analysis information not available
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Count types of scans
        if 'scan_type' in scans_df.columns:
            most_common_type = scans_df['scan_type'].mode()[0] if not scans_df.empty else 'Unknown'
            st.markdown(f"""
            <div class="health-metric">
                <div class="health-metric-title">Common Scan Type</div>
                <div class="health-metric-value">{most_common_type}</div>
                <div class="health-metric-description">
                    {sum(scans_df['scan_type'] == most_common_type)} scans of this type
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="health-metric">
                <div class="health-metric-title">Scan Types</div>
                <div class="health-metric-value">N/A</div>
                <div class="health-metric-description">
                    Scan type information not available
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display scan history
    st.markdown("<div class='section-title'>Scan History</div>", unsafe_allow_html=True)
    
    for i, scan in enumerate(scans):
        # Format scan card
        try:
            scan_date = datetime.strptime(scan['formatted_date'], "%Y-%m-%d").date() if 'formatted_date' in scan else "Unknown date"
            scan_type = scan.get('scan_type', 'Unknown type')
            scan_notes = scan.get('scan_notes', 'No notes available')
            prediction = scan.get('prediction', 'Not analyzed')
            
            # Create a status indicator based on prediction if available
            status_html = ""
            if prediction and prediction != 'Not analyzed':
                if "Non" in prediction or "Normal" in prediction:
                    status_class = "completed"
                    status_label = "Normal"
                elif "Mild" in prediction or "Converted" in prediction:
                    status_class = "scheduled"  # Using scheduled color (blue) for mild
                    status_label = "Mild Cognitive Impairment"
                else:
                    status_class = "cancelled"  # Using cancelled color (red) for severe
                    status_label = "Alzheimer's Disease"
                
                status_html = f'<span class="status-tag {status_class}">{status_label}</span>'
            
            st.markdown(f"""
            <div class="record-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div class="record-date">{scan_date}</div>
                    {status_html}
                </div>
                <div class="record-title">{scan_type}</div>
                <div class="record-subtitle">ID: {scan.get('scan_id', 'Unknown')}</div>
                <div class="record-content">
                    {scan_notes[:100]}{'...' if len(scan_notes) > 100 else ''}
                </div>
            """, unsafe_allow_html=True)
            
            # Add expandable section for full details
            if st.button(f"View Scan Details #{i}", key=f"view_scan_{i}"):
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; background-color: #f3f4f6; border-radius: 0.5rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">Scan Notes</div>
                    <div style="white-space: pre-line;">{scan_notes}</div>
                    
                    <div style="margin-top: 1rem;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">File Information</div>
                        <div>Filename: {scan.get('file_name', 'Unknown')}</div>
                        <div>Format: {scan.get('file_path', 'Unknown').split('.')[-1] if scan.get('file_path') else 'Unknown'}</div>
                    </div>
                    
                    {'<div style="margin-top: 1rem;"><div style="font-weight: 600; margin-bottom: 0.5rem;">Analysis Results</div><div>Prediction: ' + scan.get('prediction', 'Not analyzed') + '</div><div>Confidence: ' + str(round(float(scan.get('confidence', 0)) * 100, 1)) + '%</div></div>' if scan.get('prediction') else ''}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying scan: {e}")

def cognitive_assessments_tab(patient_id):
    """Display cognitive assessment results with visualizations."""
    
    # Get cognitive assessment scores
    scores = get_patient_cognitive_scores(patient_id)
    
    if not scores:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧩</div>
            <h3>No Cognitive Assessments Found</h3>
            <p>Your cognitive assessment results will appear here after your first evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Convert to dataframe for visualization
    scores_df = pd.DataFrame(scores)
    
    # Display statistics
    st.markdown("<div class='section-title'>Cognitive Assessment Overview</div>", unsafe_allow_html=True)
    
    if len(scores) > 0:
        # Get the most recent assessment
        latest = scores[0]
        
        # Show latest result in a prominent card
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Determine status based on prediction
            prediction = latest.get('prediction', 'Unknown')
            status_class = "good"
            if "Demented" in prediction or "Alzheimer" in prediction:
                status_class = "danger"
            elif "Converted" in prediction or "Mild" in prediction:
                status_class = "warning"
            
            confidence = float(latest.get('confidence_score', 0)) * 100 if 'confidence_score' in latest else 0
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Latest Assessment Result</div>
                <div class="record-date">{latest.get('formatted_date', 'Unknown date')}</div>
                
                <div style="margin: 1.5rem 0; display: flex; align-items: center;">
                    <div style="width: 80px; height: 80px; border-radius: 50%; background-color: {'#10b981' if status_class == 'good' else '#f59e0b' if status_class == 'warning' else '#ef4444'}; color: white; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 600;">
                        {confidence:.0f}%
                    </div>
                    <div style="margin-left: 1.5rem;">
                        <div style="font-weight: 600; font-size: 1.25rem; margin-bottom: 0.5rem;">{prediction}</div>
                        <div style="color: #6b7280; font-size: 0.875rem;">Confidence Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            try:
                # Create a gauge chart for confidence score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, 50], 'color': "#fee2e2"},
                            {'range': [50, 75], 'color': "#fef3c7"},
                            {'range': [75, 100], 'color': "#d1fae5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': confidence
                        }
                    },
                    title = {'text': "Confidence Level"}
                ))
                
                # Update layout
                fig.update_layout(
                    height=230,
                    margin=dict(l=10, r=10, t=50, b=10),
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")
    
    # Create trend visualization if multiple scores exist
    if len(scores) > 1:
        st.markdown("<div class='section-title'>Cognitive Health Trends</div>", unsafe_allow_html=True)
        
        try:
            # Extract dates and predictions
            dates = [datetime.strptime(score['formatted_date'], "%Y-%m-%d").date() for score in scores]
            
            # Check if we have confidence scores
            if 'confidence_score' in scores[0]:
                confidences = [float(score.get('confidence_score', 0)) * 100 for score in scores]
                
                # Create line chart
                fig = px.line(
                    x=dates, y=confidences, 
                    markers=True,
                    labels={"x": "Assessment Date", "y": "Confidence Score (%)"},
                    title="Cognitive Assessment Confidence Trends",
                    template="plotly_white",
                    color_discrete_sequence=['#3b82f6']
                )
                
                # Add annotations for specific points
                for i, conf in enumerate(confidences):
                    prediction = scores[i].get('prediction', 'Unknown')
                    fig.add_annotation(
                        x=dates[i],
                        y=conf,
                        text=prediction,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        ax=0,
                        ay=-40
                    )
                
                # Update layout
                fig.update_layout(
                    height=400,
                    hovermode="x unified",
                    xaxis=dict(tickangle=-45),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Check if we have any numeric cognitive features
            cognitive_features = [
                'MMSE', 'CDRSB', 'ADAS11', 'ADAS13', 'Hippocampus', 
                'Entorhinal', 'RAVLT_immediate', 'RAVLT_learning'
            ]
            
            available_features = []
            for feature in cognitive_features:
                if feature in scores[0]:
                    available_features.append(feature)
            
            if available_features:
                # Let user select features to plot
                selected_features = st.multiselect(
                    "Select cognitive features to view:",
                    available_features,
                    default=available_features[:3]  # Default to first 3
                )
                
                if selected_features:
                    # Create dataframe
                    data = {
                        'Date': dates
                    }
                    
                    for feature in selected_features:
                        data[feature] = [float(score.get(feature, 0)) for score in scores]
                    
                    df = pd.DataFrame(data)
                    
                    # Create line chart
                    fig = px.line(
                        df, x='Date', y=selected_features,
                        markers=True,
                        labels={"value": "Score", "variable": "Feature"},
                        title="Cognitive Feature Trends",
                        template="plotly_white"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=400,
                        hovermode="x unified",
                        xaxis=dict(tickangle=-45),
                        yaxis=dict(title="Score"),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add feature descriptions
                    st.markdown("<div class='section-title'>Feature Explanations</div>", unsafe_allow_html=True)
                    
                    feature_descriptions = {
                        'MMSE': "Mini-Mental State Examination (0-30) - Higher scores indicate better cognitive function",
                        'CDRSB': "Clinical Dementia Rating Sum of Boxes (0-18) - Higher scores indicate more impairment",
                        'ADAS11': "Alzheimer's Disease Assessment Scale (0-70) - Higher scores indicate more impairment",
                        'ADAS13': "Alzheimer's Disease Assessment Scale expanded (0-85) - Higher scores indicate more impairment",
                        'Hippocampus': "Hippocampal volume (mm³) - Lower volumes may indicate atrophy",
                        'Entorhinal': "Entorhinal cortex volume (mm³) - Lower volumes may indicate atrophy",
                        'RAVLT_immediate': "Rey Auditory Verbal Learning Test immediate recall - Higher scores indicate better memory",
                        'RAVLT_learning': "Rey Auditory Verbal Learning Test learning - Higher scores indicate better learning ability"
                    }
                    
                    for feature in selected_features:
                        description = feature_descriptions.get(feature, "No description available")
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem; padding: 0.5rem 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                            <div style="font-weight: 600; margin-bottom: 0.25rem;">{feature}</div>
                            <div style="font-size: 0.875rem; color: #6b7280;">{description}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
    
    # Assessment history
    st.markdown("<div class='section-title'>Assessment History</div>", unsafe_allow_html=True)
    
    for i, score in enumerate(scores):
        assessment_date = score.get('formatted_date', 'Unknown date')
        prediction = score.get('prediction', 'Unknown')
        confidence = float(score.get('confidence_score', 0)) * 100 if 'confidence_score' in score else 0
        
        # Determine status based on prediction
        status_class = "completed"
        if "Demented" in prediction or "Alzheimer" in prediction:
            status_class = "cancelled"
        elif "Converted" in prediction or "Mild" in prediction:
            status_class = "scheduled"
        
        st.markdown(f"""
        <div class="record-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div class="record-date">{assessment_date}</div>
                <span class="status-tag {status_class}">{prediction}</span>
            </div>
            <div class="record-title">Cognitive Assessment #{i+1}</div>
            <div class="record-subtitle">Confidence: {confidence:.1f}%</div>
            <div class="record-content">
                Analysis ID: {score.get('analysis_id', 'Unknown')}
            </div>
        </div>
        """, unsafe_allow_html=True) 