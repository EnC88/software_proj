#!/usr/bin/env python3
"""
Analytics Dashboard for Feedback System
Interactive visualizations and analytics using Streamlit and Plotly.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Define repo root for robust file access
REPO_ROOT = Path(__file__).resolve().parents[2]

# Add src to path for imports
sys.path.append(str(REPO_ROOT))

from src.evaluation.feedback_system import FeedbackLogger, FeedbackIntegration
from src.evaluation.feedback_loop import run_feedback_loop

# Configure page
st.set_page_config(
    page_title="Feedback Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize feedback system
@st.cache_resource
def get_feedback_system():
    """Get cached feedback system instance."""
    return FeedbackLogger()

def load_feedback_data():
    """Load feedback data from database."""
    try:
        feedback_logger = get_feedback_system()
        feedback_data = feedback_logger.get_all_feedback()
        
        if not feedback_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(feedback_data)
        
        # Convert timestamp to datetime with error handling
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['created_at'] = pd.to_datetime(df['created_at'])
        except Exception as e:
            st.error(f"Error parsing timestamps: {e}")
            return pd.DataFrame()
        
        # Add date columns for analysis
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        
        return df
    except Exception as e:
        st.error(f"Error loading feedback data: {e}")
        return pd.DataFrame()

def validate_date_range(start_date, end_date):
    """Validate date range input."""
    if not start_date or not end_date:
        return False, "Both start and end dates are required"
    if start_date >= end_date:
        return False, "Start date must be before end date"
    return True, None

def main():
    st.title("📊 Feedback Analytics Dashboard")
    st.markdown("---")
    
    # Add feedback loop retrain button
    if st.button("🔄 Retrain Model from Feedback"):
        with st.spinner("Retraining model and rebuilding index from feedback..."):
            result = run_feedback_loop()
        st.success(f"Retraining complete! Total feedback: {result['total_feedback']}, Negatives: {result['negative_feedback']}")
        st.balloons()
    
    # Load data
    df = load_feedback_data()
    
    if df.empty:
        st.warning("No feedback data available. Start using the system to see analytics!")
        return
    
    # Sidebar filters
    st.sidebar.header("📋 Filters")
    
    # Date range filter with validation
    try:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
        
        # Validate date range
        if len(date_range) == 2:
            is_valid, error_msg = validate_date_range(date_range[0], date_range[1])
            if not is_valid:
                st.sidebar.error(error_msg)
                return
    except Exception as e:
        st.sidebar.error(f"Error with date range: {e}")
        return
    
    # OS filter
    try:
        os_options = ['All'] + list(df['user_os'].dropna().unique())
        selected_os = st.sidebar.selectbox("Operating System", os_options)
    except Exception as e:
        st.sidebar.error(f"Error with OS filter: {e}")
        return
    
    # Score filter
    try:
        score_options = ['All', 'Positive (1)', 'Negative (0)', 'Unrated (-1)']
        selected_score = st.sidebar.selectbox("Feedback Score", score_options)
    except Exception as e:
        st.sidebar.error(f"Error with score filter: {e}")
        return
    
    # Apply filters with error handling
    try:
        filtered_df = df.copy()
        
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'] >= date_range[0]) & 
                (filtered_df['date'] <= date_range[1])
            ]
        
        if selected_os != 'All':
            filtered_df = filtered_df[filtered_df['user_os'] == selected_os]
        
        if selected_score != 'All':
            score_map = {'Positive (1)': 1, 'Negative (0)': 0, 'Unrated (-1)': -1}
            filtered_df = filtered_df[filtered_df['feedback_score'] == score_map[selected_score]]
    except Exception as e:
        st.error(f"Error applying filters: {e}")
        return
    
    # Main metrics with error handling
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_feedback = len(filtered_df)
            st.metric("Total Feedback", total_feedback)
        
        with col2:
            positive_feedback = len(filtered_df[filtered_df['feedback_score'] == 1])
            st.metric("Positive Feedback", positive_feedback)
        
        with col3:
            negative_feedback = len(filtered_df[filtered_df['feedback_score'] == 0])
            st.metric("Negative Feedback", negative_feedback)
        
        with col4:
            positive_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
            st.metric("Positive Rate", f"{positive_rate:.1f}%")
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return
    
    st.markdown("---")
    
    # Charts with error handling
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Feedback Over Time")
            
            # Daily feedback trend
            if not filtered_df.empty:
                daily_feedback = filtered_df.groupby('date').size().reset_index(name='count')
                fig_trend = px.line(
                    daily_feedback, 
                    x='date', 
                    y='count',
                    title="Daily Feedback Volume"
                )
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Feedback score distribution
                score_counts = filtered_df['feedback_score'].value_counts().reset_index()
                score_counts['score_label'] = score_counts['index'].map({
                    1: 'Positive', 0: 'Negative', -1: 'Unrated'
                })
                
                fig_pie = px.pie(
                    score_counts,
                    values='feedback_score',
                    names='score_label',
                    title="Feedback Score Distribution"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data available for selected filters")
        
        with col2:
            st.subheader("🕒 Activity Patterns")
            
            if not filtered_df.empty:
                # Hourly activity
                hourly_activity = filtered_df.groupby('hour').size().reset_index(name='count')
                fig_hourly = px.bar(
                    hourly_activity,
                    x='hour',
                    y='count',
                    title="Hourly Activity Pattern"
                )
                fig_hourly.update_layout(height=400)
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Day of week activity
                dow_activity = filtered_df.groupby('day_of_week').size().reset_index(name='count')
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_activity['day_of_week'] = pd.Categorical(dow_activity['day_of_week'], categories=dow_order, ordered=True)
                dow_activity = dow_activity.sort_values('day_of_week')
                
                fig_dow = px.bar(
                    dow_activity,
                    x='day_of_week',
                    y='count',
                    title="Day of Week Activity"
                )
                fig_dow.update_layout(height=400)
                st.plotly_chart(fig_dow, use_container_width=True)
            else:
                st.info("No data available for selected filters")
    except Exception as e:
        st.error(f"Error creating charts: {e}")
        return
    
    st.markdown("---")
    
    # OS Analysis with error handling
    try:
        st.subheader("💻 Operating System Analysis")
        
        if not filtered_df.empty:
            os_analysis = filtered_df.groupby('user_os').agg({
                'feedback_score': ['count', lambda x: (x == 1).sum(), lambda x: (x == 0).sum()]
            }).round(2)
            
            os_analysis.columns = ['Total', 'Positive', 'Negative']
            os_analysis['Positive Rate'] = (os_analysis['Positive'] / os_analysis['Total'] * 100).round(1)
            
            st.dataframe(os_analysis, use_container_width=True)
        else:
            st.info("No data available for OS analysis")
    except Exception as e:
        st.error(f"Error in OS analysis: {e}")
    
    # Session Analysis with error handling
    try:
        st.subheader("🔍 Session Analysis")
        
        if not filtered_df.empty:
            session_stats = filtered_df.groupby('session_id').agg({
                'feedback_score': ['count', lambda x: (x == 1).sum(), lambda x: (x == 0).sum()]
            }).round(2)
            
            session_stats.columns = ['Queries', 'Positive', 'Negative']
            session_stats['Positive Rate'] = (session_stats['Positive'] / session_stats['Queries'] * 100).round(1)
            session_stats = session_stats.sort_values('Queries', ascending=False)
            
            st.dataframe(session_stats.head(10), use_container_width=True)
        else:
            st.info("No data available for session analysis")
    except Exception as e:
        st.error(f"Error in session analysis: {e}")
    
    # Recent Feedback with error handling
    try:
        st.subheader("🕐 Recent Feedback")
        
        if not filtered_df.empty:
            recent_feedback = filtered_df[['timestamp', 'query', 'feedback_score', 'user_os', 'notes']].head(20)
            recent_feedback['timestamp'] = recent_feedback['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            recent_feedback['score_label'] = recent_feedback['feedback_score'].map({
                1: '✅ Positive', 0: '❌ Negative', -1: '⏳ Unrated'
            })
            
            st.dataframe(recent_feedback, use_container_width=True)
        else:
            st.info("No recent feedback available")
    except Exception as e:
        st.error(f"Error displaying recent feedback: {e}")
    
    # Export functionality with error handling
    st.markdown("---")
    st.subheader("📤 Export Data")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Filtered Data (CSV)"):
                if not filtered_df.empty:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"feedback_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
        
        with col2:
            if st.button("Export All Data (JSON)"):
                if not filtered_df.empty:
                    json_data = filtered_df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No data to export")
    except Exception as e:
        st.error(f"Error with export functionality: {e}")

if __name__ == "__main__":
    main() 