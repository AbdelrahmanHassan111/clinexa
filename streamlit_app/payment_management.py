import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import time

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "smart_clinic"
}

def get_db_connection():
    """Create a direct database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def manage_payments_and_invoices():
    """Display and manage payments and invoices section."""
    st.subheader("💰 Payments and Invoices")
    
    # Tabs for Invoices and Payments
    tab1, tab2 = st.tabs(["📃 Invoices", "💵 Payments"])
    
    # INVOICES TAB
    with tab1:
        st.subheader("Invoice Management")
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            st.error("❌ Failed to connect to the database.")
            return
        cursor = conn.cursor(dictionary=True)
        
        # Get all invoices with patient and doctor names
        try:
            cursor.execute("""
                SELECT i.invoice_id, p.full_name as patient_name, d.full_name as doctor_name, 
                       i.created_at, i.due_date, i.total_amount, i.status
                FROM invoices i
                JOIN patients p ON i.patient_id = p.patient_id
                JOIN doctors d ON i.doctor_id = d.doctor_id
                ORDER BY i.created_at DESC
            """)
            invoices = cursor.fetchall()
            
            if invoices:
                # Convert to DataFrame
                df_invoices = pd.DataFrame(invoices)
                
                # Format dates and amounts
                if 'created_at' in df_invoices.columns:
                    df_invoices['created_at'] = pd.to_datetime(df_invoices['created_at']).dt.strftime('%Y-%m-%d')
                if 'due_date' in df_invoices.columns:
                    df_invoices['due_date'] = pd.to_datetime(df_invoices['due_date']).dt.strftime('%Y-%m-%d')
                if 'total_amount' in df_invoices.columns:
                    df_invoices['total_amount'] = df_invoices['total_amount'].apply(lambda x: f"${x:.2f}")
                
                # Filter by status
                status_filter = st.multiselect(
                    "Filter by status", 
                    options=["Pending", "Paid", "Cancelled", "Overdue"],
                    default=["Pending", "Overdue"]
                )
                
                if status_filter:
                    filtered_df = df_invoices[df_invoices['status'].isin(status_filter)]
                else:
                    filtered_df = df_invoices
                
                # Display filtered invoices
                st.dataframe(filtered_df, use_container_width=True)
                
                # View invoice details
                if not filtered_df.empty:
                    invoice_id = st.selectbox(
                        "Select invoice to view details", 
                        options=filtered_df['invoice_id'].tolist(),
                        format_func=lambda x: f"Invoice #{x} - {filtered_df[filtered_df['invoice_id']==x]['patient_name'].iloc[0]}"
                    )
                    
                    if invoice_id:
                        # Get invoice items
                        cursor.execute("""
                            SELECT item_id, description, item_type, quantity, unit_price, amount
                            FROM invoice_items
                            WHERE invoice_id = %s
                        """, (invoice_id,))
                        items = cursor.fetchall()
                        
                        # Get payments for this invoice
                        cursor.execute("""
                            SELECT payment_id, payment_date, amount, payment_method, reference_number
                            FROM payments
                            WHERE invoice_id = %s
                            ORDER BY payment_date DESC
                        """, (invoice_id,))
                        payments = cursor.fetchall()
                        
                        # Display invoice details in expander
                        with st.expander("📋 Invoice Details", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            invoice_data = filtered_df[filtered_df['invoice_id'] == invoice_id].iloc[0]
                            with col1:
                                st.markdown(f"**Patient:** {invoice_data['patient_name']}")
                                st.markdown(f"**Doctor:** {invoice_data['doctor_name']}")
                            
                            with col2:
                                st.markdown(f"**Date:** {invoice_data['created_at']}")
                                st.markdown(f"**Due Date:** {invoice_data['due_date']}")
                            
                            with col3:
                                st.markdown(f"**Amount:** {invoice_data['total_amount']}")
                                st.markdown(f"**Status:** {invoice_data['status']}")
                            
                            # Display invoice items
                            st.markdown("#### Invoice Items")
                            if items:
                                df_items = pd.DataFrame(items)
                                # Format currency
                                if 'unit_price' in df_items.columns:
                                    df_items['unit_price'] = df_items['unit_price'].apply(lambda x: f"${x:.2f}")
                                if 'amount' in df_items.columns:
                                    df_items['amount'] = df_items['amount'].apply(lambda x: f"${x:.2f}")
                                st.dataframe(df_items, use_container_width=True)
                            else:
                                st.info("No items found for this invoice.")
                            
                            # Display payments
                            st.markdown("#### Payments")
                            if payments:
                                df_payments = pd.DataFrame(payments)
                                # Format dates and amounts
                                if 'payment_date' in df_payments.columns:
                                    df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date']).dt.strftime('%Y-%m-%d')
                                if 'amount' in df_payments.columns:
                                    df_payments['amount'] = df_payments['amount'].apply(lambda x: f"${x:.2f}")
                                st.dataframe(df_payments, use_container_width=True)
                            else:
                                st.info("No payments recorded for this invoice.")
                            
                            # Update invoice status
                            new_status = st.selectbox(
                                "Update Invoice Status",
                                options=["Pending", "Paid", "Cancelled", "Overdue"],
                                index=["Pending", "Paid", "Cancelled", "Overdue"].index(invoice_data['status'])
                            )
                            
                            # Move status update to its own form
                            with st.form(key="update_status_form"):
                                status_update = st.selectbox(
                                    "Update Invoice Status",
                                    options=["Pending", "Paid", "Cancelled", "Overdue"],
                                    index=["Pending", "Paid", "Cancelled", "Overdue"].index(invoice_data['status']),
                                    key="status_update_selectbox"
                                )
                                
                                status_submit = st.form_submit_button("Update Status")
                                if status_submit:
                                    try:
                                        cursor.execute(
                                            "UPDATE invoices SET status = %s WHERE invoice_id = %s",
                                            (status_update, invoice_id)
                                        )
                                        conn.commit()
                                        st.success(f"✅ Invoice status updated to {status_update}")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating invoice status: {e}")
                            
                            # Add payment button
                            st.markdown("#### Add Payment")
                            with st.form("add_payment_form"):
                                payment_amount = st.number_input("Payment Amount", min_value=0.01, format="%.2f")
                                payment_method = st.selectbox(
                                    "Payment Method", 
                                    options=["Cash", "Credit Card", "Bank Transfer", "Insurance", "Other"]
                                )
                                reference = st.text_input("Reference Number (optional)")
                                payment_date = st.date_input("Payment Date", value=datetime.now().date())
                                notes = st.text_area("Notes (optional)")
                                
                                submitted = st.form_submit_button("Record Payment")
                                if submitted:
                                    try:
                                        # Get current user ID
                                        user_id = st.session_state.get("user_id", 1)
                                        
                                        cursor.execute("""
                                            INSERT INTO payments 
                                            (invoice_id, payment_date, amount, payment_method, 
                                             reference_number, notes, created_by)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                                        """, (
                                            invoice_id, 
                                            payment_date, 
                                            payment_amount, 
                                            payment_method, 
                                            reference, 
                                            notes, 
                                            user_id
                                        ))
                                        
                                        # Check if this payment completes the invoice
                                        cursor.execute("""
                                            SELECT 
                                                i.total_amount, 
                                                COALESCE(SUM(p.amount), 0) as paid_amount
                                            FROM invoices i
                                            LEFT JOIN payments p ON i.invoice_id = p.invoice_id
                                            WHERE i.invoice_id = %s
                                            GROUP BY i.invoice_id, i.total_amount
                                        """, (invoice_id,))
                                        
                                        payment_summary = cursor.fetchone()
                                        if payment_summary and (payment_summary['paid_amount'] >= payment_summary['total_amount']):
                                            # Update invoice to paid if payments >= total amount
                                            cursor.execute(
                                                "UPDATE invoices SET status = 'Paid' WHERE invoice_id = %s",
                                                (invoice_id,)
                                            )
                                        
                                        conn.commit()
                                        st.success("✅ Payment recorded successfully")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error recording payment: {e}")
            else:
                st.info("No invoices found in the system.")
            
            # Create new invoice section
            st.subheader("Create New Invoice")
            with st.form("create_invoice_form"):
                # Get patients list
                cursor.execute("SELECT patient_id, full_name FROM patients ORDER BY full_name")
                patients = cursor.fetchall()
                
                # Get doctors list
                cursor.execute("SELECT doctor_id, full_name FROM doctors ORDER BY full_name")
                doctors = cursor.fetchall()
                
                if not patients:
                    st.error("No patients in the system. Please add patients first.")
                    st.stop()
                
                if not doctors:
                    st.error("No doctors in the system. Please add doctors first.")
                    st.stop()
                
                # Create dictionaries for selection options
                patient_options = {p['full_name']: p['patient_id'] for p in patients}
                doctor_options = {d['full_name']: d['doctor_id'] for d in doctors}
                
                # Invoice form fields
                selected_patient = st.selectbox("Patient", options=list(patient_options.keys()))
                selected_doctor = st.selectbox("Doctor", options=list(doctor_options.keys()))
                
                # Due date defaulting to 30 days from now
                due_date = st.date_input("Due Date", value=datetime.now().date() + timedelta(days=30))
                
                # Dynamic invoice items
                st.markdown("#### Invoice Items")
                
                # Initialize session state for items if not exists
                if 'invoice_items' not in st.session_state:
                    st.session_state.invoice_items = [
                        {"description": "", "item_type": "Consultation", "quantity": 1, "unit_price": 0.0}
                    ]
                
                # Display existing items
                total_amount = 0
                for i, item in enumerate(st.session_state.invoice_items):
                    cols = st.columns([3, 2, 1, 1, 1])
                    
                    with cols[0]:
                        item["description"] = st.text_input(
                            "Description", 
                            value=item["description"], 
                            key=f"desc_{i}"
                        )
                    
                    with cols[1]:
                        item["item_type"] = st.selectbox(
                            "Type", 
                            options=["Consultation", "Analysis", "MRI", "Medication", "Other"],
                            index=["Consultation", "Analysis", "MRI", "Medication", "Other"].index(item["item_type"]),
                            key=f"type_{i}"
                        )
                    
                    with cols[2]:
                        item["quantity"] = st.number_input(
                            "Qty", 
                            min_value=1, 
                            value=item["quantity"], 
                            key=f"qty_{i}"
                        )
                    
                    with cols[3]:
                        item["unit_price"] = st.number_input(
                            "Price", 
                            min_value=0.0, 
                            value=item["unit_price"], 
                            format="%.2f", 
                            key=f"price_{i}"
                        )
                    
                    with cols[4]:
                        item_amount = item["quantity"] * item["unit_price"]
                        st.text_input("Amount", value=f"${item_amount:.2f}", disabled=True, key=f"amount_{i}")
                        total_amount += item_amount
                
                # Add/remove item buttons
                col1, col2 = st.columns(2)
                
                # Display total
                st.markdown(f"**Total Amount: ${total_amount:.2f}**")
                
                # Notes field
                notes = st.text_area("Invoice Notes")
                
                # Submit button with options to add/remove items
                add_item = st.form_submit_button("Create Invoice + Add Item")
                remove_item = st.form_submit_button("Create Invoice + Remove Item")
                submitted = st.form_submit_button("Create Invoice")
                
                # Process form submission
                if add_item:
                    # Add a new item and don't submit the form
                    st.session_state.invoice_items.append(
                        {"description": "", "item_type": "Consultation", "quantity": 1, "unit_price": 0.0}
                    )
                    st.rerun()
                elif remove_item and len(st.session_state.invoice_items) > 1:
                    # Remove the last item and don't submit the form
                    st.session_state.invoice_items.pop()
                    st.rerun()
                elif submitted:
                    if selected_patient and selected_doctor and total_amount > 0:
                        try:
                            # Get patient and doctor IDs
                            patient_id = patient_options[selected_patient]
                            doctor_id = doctor_options[selected_doctor]
                            
                            # Start transaction
                            conn.start_transaction()
                            
                            # Insert invoice
                            cursor.execute("""
                                INSERT INTO invoices 
                                (patient_id, doctor_id, created_at, due_date, total_amount, status, notes)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                patient_id,
                                doctor_id,
                                datetime.now(),
                                due_date,
                                total_amount,
                                "Pending",
                                notes
                            ))
                            
                            # Get the new invoice ID
                            invoice_id = cursor.lastrowid
                            
                            # Insert invoice items
                            for item in st.session_state.invoice_items:
                                item_amount = item["quantity"] * item["unit_price"]
                                cursor.execute("""
                                    INSERT INTO invoice_items
                                    (invoice_id, description, item_type, quantity, unit_price, amount)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                """, (
                                    invoice_id,
                                    item["description"],
                                    item["item_type"],
                                    item["quantity"],
                                    item["unit_price"],
                                    item_amount
                                ))
                            
                            # Commit the transaction
                            conn.commit()
                            
                            # Clear the items from session state
                            st.session_state.invoice_items = [
                                {"description": "", "item_type": "Consultation", "quantity": 1, "unit_price": 0.0}
                            ]
                            
                            st.success(f"✅ Invoice #{invoice_id} created successfully")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            conn.rollback()
                            st.error(f"Error creating invoice: {e}")
                    else:
                        st.warning("Please fill in all required fields and add at least one item.")
        except Exception as e:
            st.error(f"Error retrieving invoices: {e}")
        finally:
            cursor.close()
            conn.close()
    
    # PAYMENTS TAB
    with tab2:
        st.subheader("Payment History")
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            st.error("❌ Failed to connect to the database.")
            return
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get all payments with related information
            cursor.execute("""
                SELECT 
                    p.payment_id, p.payment_date, p.amount, p.payment_method, 
                    p.reference_number, i.invoice_id, pat.full_name as patient_name
                FROM payments p
                JOIN invoices i ON p.invoice_id = i.invoice_id
                JOIN patients pat ON i.patient_id = pat.patient_id
                ORDER BY p.payment_date DESC
            """)
            payments = cursor.fetchall()
            
            if payments:
                df_payments = pd.DataFrame(payments)
                
                # Format dates and amounts
                if 'payment_date' in df_payments.columns:
                    df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date']).dt.strftime('%Y-%m-%d')
                if 'amount' in df_payments.columns:
                    df_payments['amount'] = df_payments['amount'].apply(lambda x: f"${x:.2f}")
                
                # Filter by payment method
                method_filter = st.multiselect(
                    "Filter by payment method", 
                    options=["Cash", "Credit Card", "Bank Transfer", "Insurance", "Other"]
                )
                
                if method_filter:
                    filtered_payments = df_payments[df_payments['payment_method'].isin(method_filter)]
                else:
                    filtered_payments = df_payments
                
                # Display payments
                st.dataframe(filtered_payments, use_container_width=True)
                
                # Payment analytics
                st.subheader("Payment Analytics")
                
                # Create a simple pie chart for payment methods
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Get payment summary by method - moved here to avoid potential form issues
                try:
                    cursor.execute("""
                        SELECT 
                            payment_method, 
                            COUNT(*) as count, 
                            SUM(amount) as total_amount
                        FROM payments
                        GROUP BY payment_method
                        ORDER BY total_amount DESC
                    """)
                    method_summary = cursor.fetchall()
                    
                    if method_summary:
                        df_summary = pd.DataFrame(method_summary)
                        
                        # Format amount
                        if 'total_amount' in df_summary.columns:
                            df_summary['total_amount'] = df_summary['total_amount'].apply(lambda x: f"${x:.2f}")
                        
                        # Display summary
                        st.markdown("#### Payments by Method")
                        st.dataframe(df_summary, use_container_width=True)
                        
                        # Convert back to numeric for plotting
                        method_summary_numeric = pd.DataFrame(method_summary)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.pie(
                            method_summary_numeric['total_amount'], 
                            labels=method_summary_numeric['payment_method'],
                            autopct='%1.1f%%',
                            startangle=90
                        )
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                        st.pyplot(fig)
                    
                    # Monthly payment trends
                    cursor.execute("""
                        SELECT 
                            DATE_FORMAT(payment_date, '%Y-%m') as month,
                            SUM(amount) as monthly_total
                        FROM payments
                        GROUP BY DATE_FORMAT(payment_date, '%Y-%m')
                        ORDER BY month
                    """)
                    monthly_trends = cursor.fetchall()
                    
                    if monthly_trends:
                        df_trends = pd.DataFrame(monthly_trends)
                        
                        # Create a line chart for monthly trends
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df_trends['month'], df_trends['monthly_total'], marker='o')
                        ax.set_xlabel('Month')
                        ax.set_ylabel('Total Payments ($)')
                        ax.set_title('Monthly Payment Trends')
                        ax.grid(True)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating payment analytics: {e}")
            else:
                st.info("No payment records found.")
        except Exception as e:
            st.error(f"Error retrieving payment data: {e}")
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    # This allows testing this module directly
    # Don't set page config here since it should be in app.py
    manage_payments_and_invoices() 