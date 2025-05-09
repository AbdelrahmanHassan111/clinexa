import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.express as px
from ..utils.db import (
    get_patient_appointments, 
    get_available_doctors, 
    schedule_new_appointment,
    update_appointment,
    get_doctor_appointments,
    get_patient_appointments_by_date
)

def schedule_appointment_page(patient_id):
    """Display appointment scheduling with enhanced UI."""
    
    # Page header
    st.markdown("""
    <div class="dashboard-header">
        <h1>Appointments</h1>
        <p>Schedule new appointments and manage your existing ones</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for appointment management
    tab1, tab2 = st.tabs(["My Appointments", "Schedule New"])
    
    # My Appointments Tab
    with tab1:
        my_appointments_tab(patient_id)
    
    # Schedule New Tab
    with tab2:
        schedule_new_tab(patient_id)

def my_appointments_tab(patient_id):
    """Display existing appointments with actions."""
    
    # Get patient appointments
    appointments = get_patient_appointments(patient_id)
    
    if not appointments:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📅</div>
            <h3>No Appointments Found</h3>
            <p>You don't have any scheduled appointments. Use the "Schedule New" tab to book your first appointment.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "Scheduled", "Completed", "Cancelled"],
            index=0
        )
    
    with col2:
        time_filter = st.selectbox(
            "Time period",
            ["All time", "Upcoming", "Past", "Next 30 days", "Last 30 days"],
            index=1
        )
    
    # Apply filters
    filtered_appointments = filter_appointments(appointments, status_filter, time_filter)
    
    if not filtered_appointments:
        st.info("No appointments match your filter criteria.")
        return
    
    # Upcoming appointments section
    upcoming = [a for a in filtered_appointments 
                if datetime.strptime(a['formatted_date'], "%Y-%m-%d").date() >= date.today() 
                and a['status'] == 'scheduled']
    
    if upcoming:
        st.markdown("<div class='section-title'>Upcoming Appointments</div>", unsafe_allow_html=True)
        for i, appt in enumerate(upcoming):
            display_appointment(appt, i, True)
    
    # Past appointments section
    past = [a for a in filtered_appointments 
           if datetime.strptime(a['formatted_date'], "%Y-%m-%d").date() < date.today() 
           or a['status'] in ['completed', 'cancelled']]
    
    if past:
        st.markdown("<div class='section-title'>Past Appointments</div>", unsafe_allow_html=True)
        for i, appt in enumerate(past):
            display_appointment(appt, i + len(upcoming), False)
    
    # Appointment statistics
    if status_filter == "All" and time_filter == "All time":
        st.markdown("<div class='section-title'>Appointment Stats</div>", unsafe_allow_html=True)
        display_appointment_stats(appointments)

def display_appointment(appt, index, is_upcoming):
    """Display a single appointment with actions."""
    # Format date and time
    try:
        appt_date = datetime.strptime(appt['formatted_date'], "%Y-%m-%d").date()
        date_display = appt_date.strftime("%A, %B %d, %Y")
        
        time_str = appt['formatted_time'] if 'formatted_time' in appt else "Unknown time"
    except:
        date_display = appt.get('formatted_date', 'Unknown date')
        time_str = "Unknown time"
    
    doctor_name = appt.get('doctor_name', 'Unknown doctor')
    appt_type = appt.get('appointment_type', 'Consultation')
    status = appt.get('status', 'scheduled')
    appt_id = appt.get('appointment_id', 0)
    
    # Determine status class for visual indicator
    status_class = "scheduled"
    if status == "completed":
        status_class = "completed"
    elif status == "cancelled":
        status_class = "cancelled"
    
    # Create the appointment card
    st.markdown(f"""
    <div class="record-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div class="record-date">{date_display} at {time_str}</div>
            <span class="status-tag {status_class}">{status.title()}</span>
        </div>
        <div class="record-title">{appt_type}</div>
        <div class="record-subtitle">Dr. {doctor_name}</div>
        <div class="record-content">
            Appointment ID: {appt_id}
            {f"<br>Notes: {appt.get('notes', 'No notes')[:100]}{'...' if len(appt.get('notes', '')) > 100 else ''}" if appt.get('notes') else ""}
        </div>
    """, unsafe_allow_html=True)
    
    # Add actions for upcoming appointments
    if is_upcoming and status == "scheduled":
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📝 Reschedule", key=f"reschedule_{index}", use_container_width=True):
                st.session_state.reschedule_appt = appt_id
                st.rerun()
        
        with col2:
            if st.button(f"❌ Cancel", key=f"cancel_{index}", use_container_width=True):
                # Confirm cancellation
                st.warning("Are you sure you want to cancel this appointment?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Cancel Appointment", key=f"confirm_cancel_{index}", use_container_width=True):
                        # Call function to cancel appointment
                        success = update_appointment(appt_id, "cancelled", "Cancelled by patient")
                        
                        if success:
                            st.success("Appointment cancelled successfully")
                            st.rerun()
                        else:
                            st.error("Failed to cancel appointment")
                with col2:
                    if st.button("No, Keep Appointment", key=f"keep_appt_{index}", use_container_width=True):
                        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def filter_appointments(appointments, status_filter, time_filter):
    """Filter appointments based on status and time period."""
    filtered = appointments.copy()
    
    # Apply status filter
    if status_filter != "All":
        filtered = [a for a in filtered if a.get('status', '').lower() == status_filter.lower()]
    
    # Apply time filter
    today = date.today()
    if time_filter == "Upcoming":
        filtered = [a for a in filtered 
                  if datetime.strptime(a['formatted_date'], "%Y-%m-%d").date() >= today]
    elif time_filter == "Past":
        filtered = [a for a in filtered 
                  if datetime.strptime(a['formatted_date'], "%Y-%m-%d").date() < today]
    elif time_filter == "Next 30 days":
        filtered = [a for a in filtered 
                  if today <= datetime.strptime(a['formatted_date'], "%Y-%m-%d").date() <= today + timedelta(days=30)]
    elif time_filter == "Last 30 days":
        filtered = [a for a in filtered 
                  if today - timedelta(days=30) <= datetime.strptime(a['formatted_date'], "%Y-%m-%d").date() < today]
    
    return filtered

def display_appointment_stats(appointments):
    """Display statistics about patient appointments."""
    if not appointments:
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(appointments)
    
    # Add datetime column
    df['appt_date'] = df['formatted_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
    
    # Compute statistics
    total = len(df)
    completed = len(df[df['status'] == 'completed'])
    cancelled = len(df[df['status'] == 'cancelled'])
    scheduled = len(df[df['status'] == 'scheduled'])
    
    # Create status distribution pie chart
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = px.pie(
        status_counts, 
        values='Count', 
        names='Status', 
        title="Appointment Status Distribution",
        color='Status',
        color_discrete_map={
            'completed': '#10b981',
            'scheduled': '#3b82f6',
            'cancelled': '#ef4444'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    # Display in columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display key metrics
        st.markdown(f"""
        <div style="padding: 1rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
            <div style="font-weight: 600; margin-bottom: 1rem; font-size: 1.125rem;">Appointment Summary</div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <div style="color: #6b7280;">Total Appointments</div>
                <div style="font-weight: 500;">{total}</div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <div style="color: #6b7280;">Completed</div>
                <div style="font-weight: 500; color: #10b981;">{completed}</div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <div style="color: #6b7280;">Scheduled</div>
                <div style="font-weight: 500; color: #3b82f6;">{scheduled}</div>
            </div>
            
            <div style="display: flex; justify-content: space-between;">
                <div style="color: #6b7280;">Cancelled</div>
                <div style="font-weight: 500; color: #ef4444;">{cancelled}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display pie chart
        st.plotly_chart(fig, use_container_width=True)

def schedule_new_tab(patient_id):
    """Schedule a new appointment with a doctor."""
    
    st.markdown("<div class='section-title'>Schedule Your Next Appointment</div>", unsafe_allow_html=True)
    
    # Step 1: Choose appointment type
    with st.container():
        st.markdown("""
        <div class="step-container active">
            <div class="step-header">
                <div class="step-number">1</div>
                <div class="step-title">Choose Appointment Type</div>
            </div>
        """, unsafe_allow_html=True)
        
        appointment_types = [
            "Initial Consultation",
            "Follow-up Visit",
            "Cognitive Assessment",
            "MRI Scan",
            "Treatment Discussion",
            "Medication Review",
            "Other"
        ]
        
        appointment_type = st.selectbox(
            "What type of appointment do you need?",
            appointment_types,
            index=0
        )
        
        if appointment_type == "Other":
            custom_type = st.text_input("Please specify appointment type:")
            if custom_type:
                appointment_type = custom_type
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 2: Choose a doctor
    with st.container():
        st.markdown("""
        <div class="step-container">
            <div class="step-header">
                <div class="step-number">2</div>
                <div class="step-title">Select Doctor</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Get available doctors
        doctors = get_available_doctors()

        if not doctors:
            st.error("No doctors are currently available. Please try again later.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Create dropdown for doctor selection
        doctor_options = {f"Dr. {doc['full_name']} - {doc['specialty']}": doc['doctor_id'] for doc in doctors}

        # Add an empty option
        doctor_options = {"Select a doctor": ""} | doctor_options

        selected_doctor_display = st.selectbox(
            "Choose your preferred doctor",
            options=list(doctor_options.keys()),
            index=0
        )

        # Get the selected doctor ID
        selected_doctor = doctor_options[selected_doctor_display]

        # Show doctor details if selected
        if selected_doctor:
            # Find the selected doctor in the list
            doctor_info = next((d for d in doctors if d['doctor_id'] == selected_doctor), None)
            if doctor_info:
                st.markdown(f"""
                <div style="padding: 15px; background-color: #f0f9ff; border-radius: 5px; margin-top: 10px;">
                    <h4 style="margin-top: 0;">Dr. {doctor_info['full_name']}</h4>
                    <p><strong>Specialty:</strong> {doctor_info.get('specialty', 'General Medicine')}</p>
                    <p><strong>Experience:</strong> {doctor_info.get('years_experience', 0)} years</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 3: Choose date and time
    with st.container():
        st.markdown("""
        <div class="step-container">
            <div class="step-header">
                <div class="step-number">3</div>
                <div class="step-title">Select Date & Time</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date selection with min/max constraints
            min_date = date.today() + timedelta(days=1)  # Start from tomorrow
            max_date = date.today() + timedelta(days=60)  # Allow booking up to 60 days ahead
            
            appointment_date = st.date_input(
                "Select a date",
                min_value=min_date,
                max_value=max_date,
                value=min_date
            )
        
        with col2:
            # Time selection based on doctor availability
            available_times = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
                              "13:00", "13:30", "14:00", "14:30", "15:00", "15:30"]
            
            # Check for existing appointments if a doctor is selected
            if selected_doctor:
                # Convert date to string
                appointment_date_str = appointment_date.strftime("%Y-%m-%d")
                
                # Get doctor's existing appointments for the selected date
                existing_appointments = get_doctor_appointments(selected_doctor, appointment_date_str)
                
                # Extract times that are already booked
                booked_times = [appt['formatted_time'] for appt in existing_appointments 
                               if appt['status'].lower() != 'cancelled']
                
                # Filter out booked times
                available_times = [t for t in available_times if t not in booked_times]
                
                if not available_times:
                    st.warning("⚠️ No available times for this doctor on the selected date. Please choose another date.")
                    # Add a full day option
                    available_times = ["No available times"]
            
            appointment_time = st.selectbox(
                "Select a time",
                available_times,
                index=0
            )
        
        # Additional notes
        appointment_notes = st.text_area(
            "Notes for the doctor (optional)",
            placeholder="Describe your symptoms or reason for visit...",
            height=100
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add availability check button
    availability_checked = False
    if 'availability_checked' not in st.session_state:
        st.session_state.availability_checked = False
    if 'is_available' not in st.session_state:
        st.session_state.is_available = False

    if st.button("Check Availability", use_container_width=True):
        if not selected_doctor:
            st.error("Please select a doctor.")
        else:
            # Format the date and time
            appointment_date_str = appointment_date.strftime("%Y-%m-%d")
            
            # Check doctor's existing appointments
            doctor_existing_appointments = get_doctor_appointments(selected_doctor, appointment_date_str)
            
            # Check patient's existing appointments
            patient_existing_appointments = get_patient_appointments_by_date(patient_id, appointment_date_str)
            
            # Check for conflicts
            doctor_conflict = False
            patient_conflict = False
            
            for existing in doctor_existing_appointments:
                if (existing['formatted_time'] == appointment_time and 
                    existing['status'].lower() == 'scheduled'):
                    doctor_conflict = True
                    st.error(f"⚠️ Doctor is already booked at {appointment_time} on {appointment_date.strftime('%A, %B %d, %Y')}.")
                    break
            
            for existing in patient_existing_appointments:
                if (existing['formatted_time'] == appointment_time and 
                    existing['status'].lower() == 'scheduled'):
                    patient_conflict = True
                    st.error(f"⚠️ You already have an appointment at {appointment_time} on {appointment_date.strftime('%A, %B %d, %Y')}.")
                    break
            
            if not doctor_conflict and not patient_conflict:
                st.success("✅ This time slot is available! You can schedule your appointment.")
                st.session_state.availability_checked = True
                st.session_state.is_available = True
            else:
                st.session_state.availability_checked = True
                st.session_state.is_available = False

    # Submit button
    if st.button("Schedule Appointment", use_container_width=True, type="primary"):
        # Validate inputs
        if not selected_doctor:
            st.error("Please select a doctor.")
            return
        
        if appointment_time == "No available times":
            st.error("Please select a different date with available appointment times.")
            return
        
        if not st.session_state.availability_checked:
            st.error("Please check availability first.")
            return
        
        if not st.session_state.is_available:
            st.error("This time slot is not available. Please select another time or date.")
            return
        
        # Format date and time
        appointment_date_str = appointment_date.strftime("%Y-%m-%d")
        
        # Schedule the appointment
        appointment_id = schedule_new_appointment(
            patient_id,
            selected_doctor,
            appointment_date_str,
            appointment_time,
            appointment_type,
            appointment_notes
        )
        
        if appointment_id:
            st.success(f"Appointment scheduled successfully! Your appointment ID is #{appointment_id}.")
            
            # Get doctor name for display
            doctor_name = next((d['full_name'] for d in doctors if d['doctor_id'] == selected_doctor), "Unknown")
            
            # Show confirmation
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1.5rem; background-color: #ecfdf5; border-radius: 0.5rem; border: 1px solid #10b981;">
                <div style="font-weight: 600; font-size: 1.25rem; margin-bottom: 1rem; color: #10b981;">Appointment Confirmed</div>
                
                <div style="margin-bottom: 0.5rem;"><strong>Date:</strong> {appointment_date.strftime("%A, %B %d, %Y")}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Time:</strong> {appointment_time}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Type:</strong> {appointment_type}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Doctor:</strong> Dr. {doctor_name}</div>
                
                <div style="margin-top: 1rem; font-size: 0.875rem; color: #374151;">
                    Please arrive 15 minutes before your scheduled time. If you need to cancel or reschedule, please do so at least
                    24 hours in advance.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset availability check
            st.session_state.availability_checked = False
            st.session_state.is_available = False
            
            # Option to view all appointments
            if st.button("View All My Appointments"):
                # Switch to My Appointments tab
                st.session_state.appointments_tab = "My Appointments"
                st.rerun()
        else:
            st.error("Failed to schedule appointment. Please try again later.") 